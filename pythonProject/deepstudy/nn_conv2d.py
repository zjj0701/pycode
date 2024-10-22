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
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("./logs")
tudui = Tudui()
step = 0
for data in dataloader:
    img, label = data
    out = tudui(img)
    print(img.shape)  # [64, 3, 32, 32]
    # [64, 6, 30, 30]
    #img = img[0]  # 解法一：这里注意 add_image 方法期望输入张量的形状为 (C, H, W)
    writer.add_image("input", img, step,dataformats="NCHW")
    out = torch.reshape(out, (-1, 3, 30, 30))
    print(out.shape)
    out = out[0]
    writer.add_image("output", out, step)
    step += 1
