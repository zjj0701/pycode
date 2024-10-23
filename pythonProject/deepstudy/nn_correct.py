import torch

outputs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

# 最大概率的位置，1是看列，0是看行
print(outputs.argmax(1))
pre = outputs.argmax(1)
target = torch.tensor([0,1])
print((pre == target).sum().item())