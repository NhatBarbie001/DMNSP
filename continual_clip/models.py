from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F
import clip.clip as clip
import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_class_ids_per_task, get_class_names
from . import utils
from .dynamic_dataset import DynamicDataset

DEFAULT_THRESHOLD = 0.985
TOP_SELECT = 1
EPOCH_NUM = 4
TOP_K_RATIO = 0.1
LAMBDA_SCALE = 30
LAYER_NUM = 12

# def intra_cls(logits, y, classes):
#     y = y - classes
#     logits1 = logits[:, classes:]
#     return F.cross_entropy(logits1, y, reduction='none')
# class VisionClassifier(nn.Module):
#     def __init__(self, in_features, num_classes, weight_init=None, activation=None):
#         super().__init__()
#         self.fc = nn.Linear(in_features, num_classes, bias=False)
#         self.fc = nn.Parameter(self.fc.weight.data)
#         if weight_init is not None:
#             self.fc.data = weight_init
#         if activation is not None:
#             self.activation = activation
#         else:
#             self.activation = nn.Identity()
    
#     def add_weight(self, weight):
#         self.fc = nn.Parameter(torch.cat([self.fc, weight], dim=0))

#     def set_weight(self, weight):
#         self.fc = nn.Parameter(weight)


#     def forward(self, x):
#         # normalize the weights
#         x = F.normalize(x, p=2, dim=-1)
#         weight = F.normalize(self.fc, p=2, dim=-1)
#         x = F.linear(x, weight)
#         x = self.activation(x)
#         return x


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, origin_flag, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.origin_flag = origin_flag
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)
        self.prev_gradients = None
        self.visual_cur_matrix = {}
        self.visual_U = {}
        self.loss_list = []

        self.visual_clsf_epochs = cfg.visual_clsf_epochs
        self.visual_clsf_batch_size = cfg.visual_clsf_batch_size
        self.vision_clsf = None
        if cfg.model_name == "ViT-L/14":
            self.vision_clsf = VisionClassifier(768, cfg.increment, activation=None)
        else:
            self.vision_clsf = VisionClassifier(512, cfg.increment, activation=None)


    def forward(self, image, taskid):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens, 0, is_train=False)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names, world):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        # Fixed here
        if task_id == 0:
            class_names = []
            for i in range(cfg.task_num):
                class_names += get_class_names(self.classes_names, self.class_ids_per_task[i])
            self.all_class_names = class_names
            self.all_text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.all_class_names]
            ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names, world)

    # Fixed hereeeeeeeeeeeeee=======================================================
    # def forward_clip(self, image, text, return_feature=False):
    #     image_features, _ = self.model.encode_image(image)
    #     text_features, _ = self.model.encode_text(text)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.model.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     if return_feature:
    #         return logits_per_image, logits_per_text, image_features, text_features
    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text
    # # Fixed hereeeeeeeeeeeeee=======================================================
    # def forward_for_extra_visual_clsf(self, image, test=False, all_test=False, return_feature=False, replay=None):
    #     if test:
    #         # pdb.set_trace()
    #         with torch.no_grad():
    #             if all_test:
    #                 if return_feature:
    #                     logits_per_image, _, image_features, __ = self.forward_clip(image, self.all_text_tokens, return_feature=return_feature)
    #                 else:
    #                     logits_per_image, _ = self.forward_clip(image, self.all_text_tokens)
    #                 # logits_per_image = self.inference(image, self.all_text_tokens)
    #             else:
    #                 if return_feature:
    #                     logits_per_image, _, image_features, __ = self.forward_clip(image, self.text_tokens, return_feature=return_feature)
    #                 else:
    #                     logits_per_image, _ = self.forward_clip(image, self.text_tokens)
    #             # pdb.set_trace()
    #             probs = logits_per_image.softmax(dim=-1)
    #     else:

    #         if return_feature:
    #             __, _, image_features, text_features = self.forward_clip(image, self.text_tokens, return_feature=return_feature)
    #             return image_features, text_features
    #         if replay is not None:
    #             logits_per_image, _ = self.forward_clip(image, self.text_tokens)
    #             # text_features_for_replay = self.model.encode_text(self.text_tokens[:-self.cfg.increment])
    #             text_features_for_replay, _ = self.model.encode_text(self.text_tokens)
    #             text_features_for_replay, _ = text_features_for_replay / text_features_for_replay.norm(dim=1, keepdim=True)
    #             replay_features = replay / replay.norm(dim=1, keepdim=True)
    #             replay_logits = replay_features @ text_features_for_replay.t() * 100
    #         else:
    #             logits_per_image, _ = self.forward_clip(image, self.text_tokens)
    #         probs = logits_per_image
                
    #     if return_feature:
    #         text_features, _ = self.model.encode_text(self.all_text_tokens)
    #         return probs, image_features, text_features

    #     if replay is not None:
    #         return probs, replay_logits
    #     return probs
    def train(self, task_id, cfg, train_dataset, train_classes_names, world):

        train_loader = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)
        # if task_id == 0:
        #     targets_bais = 0
        # else:
        #     targets_bais = cfg.initial_increment + (task_id - 1) * cfg.increment
        train_iter = iter(train_loader)
        EPOCH = EPOCH_NUM
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        print(f"\n========== Task {task_id} ==========\n")

        for k, v in self.model.named_parameters():
            if "adapt" not in k :
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() if "adapt" in k
        ]
        params_name = [
            k for k, v in self.model.named_parameters() if "adapt" in k
        ]

        # print('==================trainable params========================================', params_name)
        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )
        self.model = self.model.cuda(device=0)


        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        print(classnames)
        texts = [self.prompt_template.format(c) for c in classnames]
        texts = clip.tokenize(texts).cuda(device=0)


        self.model.train()

        batch_count = 0
        lamda = [[0 for _ in range(LAYER_NUM)] for _ in range(LAYER_NUM)]
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            inputs, targets = inputs.cuda(device=0), targets.cuda(device=0)
            logits_per_image, _ = self.model.cuda(device=0)(inputs, texts.cuda(device=0), 0, is_train=True)  # 分开

            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)
            self.loss_list.append(loss)
            print('CELoss: {}'.format(loss))

            optimizer.zero_grad()
            loss.backward()

            if task_id != 0:
                if batch_count == 0:
                    for j in range(LAYER_NUM):
                        activation_visual = self.model.visual.transformer.lora_feature[j]
                        activation_visual = torch.bmm(activation_visual.detach().permute(1, 2, 0),
                                                      activation_visual.detach().permute(1, 0, 2)).sum(dim=0)
                        U_visual, S, Vh = torch.linalg.svd(activation_visual, full_matrices=False)
                        U_visual = U_visual[:, :TOP_SELECT]

                        for k in range(LAYER_NUM):
                            v_visual = self.visual_U[k]

                            normalized_vector_visual = U_visual / torch.norm(U_visual)
                            similarities_visual = []
                            for column_visual in v_visual.t():
                                normalized_column_visual = column_visual / torch.norm(column_visual)
                                cos_sim_visual = torch.dot(normalized_vector_visual.squeeze(),
                                                           normalized_column_visual.squeeze())
                                similarities_visual.append(cos_sim_visual)

                            dot_products_visual = torch.mean(
                                torch.topk(torch.stack(similarities_visual), int(len(similarities_visual) * TOP_K_RATIO))[0])
                            lamda[j][k] = torch.exp(-dot_products_visual) * LAMBDA_SCALE

                    batch_count = batch_count + 1
                for name, params in self.model.named_parameters():

                    for i in range(LAYER_NUM):
                        if 'visual' in name and 'adapt' in name and 'down' in name and 'weight' in name:
                            v = self.visual_U[i]
                            v_ = torch.mm(params.grad.data, v)
                            params.grad.data = torch.mm(v_, v.T)* lamda[int(name.split(".")[3])][i]

                        elif 'visual' in name and 'adapt' in name and 'up' in name and 'weight' in name:
                            v = self.visual_U[i]
                            v_ = torch.mm(v.T, params.grad.data)
                            params.grad.data = torch.mm(v, v_)* lamda[int(name.split(".")[3])][i]

            optimizer.step()

        torch.cuda.empty_cache()
        # # fix here:============================================================================
        # if cfg.visual_clsf:
        #     # pdb.set_trace()
        #     torch.cuda.empty_cache()
        #     self.model.eval()
        #     e_num = cfg.visual_clsf_epochs
        #     vision_clsf_loader = DataLoader(
        #         train_dataset[task_id:task_id + 1],
        #         batch_size=self.visual_clsf_batch_size,
        #         shuffle=True,
        #         num_workers=2,
        #     )
        #     features_dict = {}
        #     with torch.no_grad():
        #         for inputs, targets, t in tqdm(vision_clsf_loader):
        #             inputs, targets = inputs.to(self.device), targets.to(self.device)
        #             _, features, __ = self.forward_for_extra_visual_clsf(inputs, test=True, return_feature=True)
        #             for feature, target in zip(features, targets):
        #                 target = target.item()
        #                 if target not in features_dict:
        #                     features_dict[target] = []
        #                 features_dict[target].append(feature.cpu())
        #     mean_features = []
        #     for target in sorted(features_dict.keys()):
        #         features = torch.stack(features_dict[target])
        #         mean_feature = features.mean(dim=0)
        #         mean_features.append(mean_feature.unsqueeze(0))
        #     mean_features = torch.cat(mean_features).to(self.device)
        #     if task_id > 0:
        #         self.vision_clsf.add_weight(mean_features)
        #         pass
        #     else:
        #         self.vision_clsf.set_weight(mean_features)
        #         pass
        #     optimizer = torch.optim.Adam(self.vision_clsf.parameters(), lr=cfg.visual_clsf_lr)
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, e_num*len(vision_clsf_loader), eta_min=cfg.visual_clsf_lr*0.01)
        #     # total_vc_batches = e_num * len(vision_clsf_loader)
            
        #     for e in range(e_num):
        #         bach_i = -1
        #         for inputs, targets, t in vision_clsf_loader:
        #             inputs, targets = inputs.to(self.device), targets.to(self.device)
        #             # pdb.set_trace()
        #             with torch.no_grad():
        #                 outputs, _ = self.forward_for_extra_visual_clsf(inputs, return_feature=True)
        #             # pdb.set_trace()
        #             outputs = self.vision_clsf(outputs)
        #             # pdb.set_trace()
        #             loss = intra_cls(outputs,targets, targets_bais).mean()
        #             # loss = F.cross_entropy(outputs, targets)
        #             optimizer.zero_grad()
        #             loss.backward()
        #             optimizer.step()
        #             bach_i+=1
        #             if bach_i % 10 == 0:
        #                 logging.info(f"Epoch {e + 1}/{e_num} | Batch {bach_i + 1}/{len(vision_clsf_loader)} | Loss: {loss.item()}")
        #             scheduler.step()
        #======================================================================================
        train_loader_ = DataLoader(train_dataset[task_id:task_id + 1],
                                  batch_size=128,
                                  shuffle=True, num_workers=8)
        counts = 0
        models = self.model.cuda(0)
        for inputs, targets, task_ids in tqdm(train_loader_):
            inputs = inputs.cuda(device=0)

            with torch.no_grad():
                outputs = models(inputs, texts.cuda(0), 0, is_train=False)


            for i in range(LAYER_NUM):
                if len(self.visual_cur_matrix) == i:
                    activation = models.visual.transformer.lora_feature[i]
                    activation = torch.bmm(activation.detach().permute(1, 2, 0),
                                           activation.detach().permute(1, 0, 2)).sum(dim=0)
                    self.visual_cur_matrix[i] = activation

                    U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                    self.visual_U[i] = U[:,TOP_SELECT:]

                else:
                    activation = models.visual.transformer.lora_feature[i]
                    activation = torch.bmm(activation.detach().permute(1, 2, 0),
                                           activation.detach().permute(1, 0, 2)).sum(dim=0)

                    U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                    Ui = torch.cat((self.visual_U[i], U1[:, TOP_SELECT:]), dim=1)
                    self.visual_U[i] = Ui

            counts = counts + 1
            if counts == 1:
                break

        torch.cuda.empty_cache()
        self.model.eval()

class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device, origin_flag) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device, origin_flag)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)