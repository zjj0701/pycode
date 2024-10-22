import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset/cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.liner = nn.Linear(196608, 10)

    def forward(self, x):
        out = self.liner(x)
        return out


tudui = Tudui()
for data in dataloader:
    img, label = data
    out = torch.flatten(img)
    output = tudui(out)
    print(output)
