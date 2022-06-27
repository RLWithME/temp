import torch

print(torch.prod(torch.Tensor([1,2,3,4])) * torch.Tensor([1,2,3,4]))
print(torch.prod(torch.Tensor([1,2,3,4])).item() * torch.Tensor([1,2,3,4]))