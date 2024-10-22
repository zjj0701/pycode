import torch
from torch import nn

i = torch.tensor([1.0, 2, 3], dtype=torch.float)
o = torch.tensor([1.0, 2, 5], dtype=torch.float)
i = torch.reshape(i, (1, 1, 3, 3))
o = torch.reshape(o, (1, 1, 3, 3))

# 计算方式为sum，默认为mean
loss = nn.L1Loss(reduction='sum')
res = loss(i, o)
print(f"L1 loss is:{res}")

loss2 = nn.MSELoss()
res2 = loss2(i, o)
print(f"MSE loss is:{res2}")


x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss3 = nn.CrossEntropyLoss()
res3 = loss3(x,y)
print(f"交叉熵 loss is:{res3}")