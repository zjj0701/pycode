import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.moudel = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.moudel(x)
        return x


loss = nn.CrossEntropyLoss()

tudui = Tudui()

# 设置优化器
opt = torch.optim.SGD(tudui.parameters(), lr=0.01, momentum=0.9)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, labels = data
        out = tudui(imgs)
        res_loss = loss(out, labels)
        opt.zero_grad()
        res_loss.backward()
        opt.step()
        running_loss += res_loss.item()
    print(running_loss)
