import torch
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("当前GPU索引:", torch.cuda.current_device())