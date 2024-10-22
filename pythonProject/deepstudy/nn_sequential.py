import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5,
                               padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5,
                               padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5,
                               padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.liner1 = nn.Linear(1024, 64)
        self.liner2 = nn.Linear(64, 10)

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


tudui = Tudui()

# 判断网络是否正确
i = torch.ones(64, 3, 32, 32)
o = tudui(i)
# print(o.shape)

# 画图
writer = SummaryWriter(log_dir='./logs/graph')
writer.add_graph(tudui, i)
writer.close()