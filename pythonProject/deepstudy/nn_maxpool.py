import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset/cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = nn.MaxPool2d(3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


writer = SummaryWriter("./logs/maxpool")
tudui = Tudui()
step = 0
for data in dataloader:
    img, label = data
    out = tudui(img)
    print(img.shape)  # [64, 3, 32, 32]
    # 解法一：这里注意 add_image 方法期望输入张量的形状为 (C, H, W)
    writer.add_image("input", img, step, dataformats="NCHW")
    writer.add_image("output", out, step, dataformats="NCHW")
    step += 1
writer.close()
