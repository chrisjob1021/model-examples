import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_properties(0))
print(torch.randn(1).cuda())