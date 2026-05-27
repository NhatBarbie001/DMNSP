import time

# Cấu hình delay cho từng loại dòng
def get_delay(line):
    if not line.strip():
        return 0.3                          # dòng trống
    if line.startswith("Model"):
        return 2
    if line.startswith("Files"):
        return 0.4
    if line.startswith("Evaluation"):
        return 2
    if "UserWarning" in line or "warnings.warn" in line:
        return 0.1
    if line.startswith("positive") or line.startswith("negative"):
        return 10
    if line.startswith("Epoch"):
        return 10                      # mô phỏng thời gian train mỗi batch
    return 0.2

def simulate(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        print(line, end="", flush=True)
        time.sleep(get_delay(line))

simulate("log.text")