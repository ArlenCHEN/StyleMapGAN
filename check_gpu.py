import torch

device_num = 6
torch.cuda.set_device(device_num)

print('In generate, device count: ', torch.cuda.device_count())
print('In generate, current device: ', torch.cuda.current_device())
