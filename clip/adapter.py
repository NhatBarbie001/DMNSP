# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
from re import X
import torch
import torch.nn as nn
from torch.nn.functional import linear

class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 text_or_image=None):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # self.linear = nn.Linear(self.n_embd, self.n_embd)

        self.down_proj = nn.Linear(self.n_embd, 64)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        elif init_option == "linear":
            with torch.no_grad():
                nn.init.zeros_(self.linear.weight)
        # ================ Fixed hereee=========================
        #=================FFT heree=============================
        #--------------FFT heree----------------
        self.text_or_image = text_or_image
        self.n_frq = 3000

        self.device = 0
        #Fix hard num tasks = 1
        self.num_tasks = 1 # cfg.num_tasks
        # 👉 tạo generator riêng
        # generator cho weight (GPU)
        g_cuda = torch.Generator(device=self.device)
        g_cuda.manual_seed(29)

        # generator cho indices (CPU)
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(11)

        self.coef_mlp = nn.ParameterList([
        nn.Parameter(torch.randn(self.n_frq, generator=g_cuda, device=self.device), requires_grad=True)
        for _ in range(self.num_tasks)
        ])
        self.image_dim = self.n_embd
        self.indices = [
        self.select_pos(t, self.image_dim, generator=g_cpu).to(self.device)
        for t in range(self.num_tasks)
        ]
        self.init_param()
        #---------------------------------------
        #================================================================
    def init_param(self):
        for t in range(len(self.coef_mlp)):
            nn.init.zeros_(self.coef_mlp[t])
    
    # ⚠️ sửa select_pos để nhận generator
    def select_pos(self, t, dim, generator=None):
        print(f"embed_dim:  {dim}")
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(777 + t * 10)

        indices = torch.randperm(dim * dim, generator=generator)[:self.n_frq]
        indices = torch.stack([indices // dim, indices % dim], dim=0)
        return indices

    def get_delta_mlp(self, task, alpha=3000):
        coef = self.coef_mlp[task]
        device = coef.device
        F = torch.zeros(self.embed_dim, self.embed_dim).to(device)
        indices = self.indices[task]
        F[indices[0,:], indices[1,:]] =  self.coef_mlp[task]

        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
    #================================================================
    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)
        # ------Fixed hereeee-------------------------------------------------------
        if "text" in self.text_or_image:
            _cur_task = 0
            
            weight_delta_mlp = torch.stack([self.get_delta_mlp(t) for t in range(_cur_task+1)], dim=0).sum(dim=0)
            up = linear(x, weight_delta_mlp)
            # weight_v = torch.stack([self.get_delta_w_v(t) for t in range(_cur_task+1)], dim=0).sum(dim=0)
        else:
            down = self.down_proj(x)
            down = self.non_linear_func(down)
            down = nn.functional.dropout(down, p=self.dropout, training=self.training)
            up = self.up_proj(down) #git push -u origin experiment_FFT_Coeficient_Parameter

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return down, output, \
            self.up_proj.weight, self.down_proj.weight, self.up_proj.bias, self.down_proj.bias