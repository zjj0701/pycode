import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset/cifar10", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root="./dataset/cifar10", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 数据集的大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集大小：{train_data_size},测试集大小:{test_data_size}")

# 加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)


# 搭建神经网络
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


tudui = Tudui()
# 1
if torch.cuda.is_available():
    tudui = tudui.cuda()



# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 2
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
learning_rate = 0.01
opt = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch_num = 10

# 添加tensorboard
writer = SummaryWriter(log_dir='./log')

for i in range(epoch_num):
    print(f"第 {i + 1} 轮训练开始")

    # 训练开始
    tudui.train()  # drop,batchnorm有作用
    for data in train_data_loader:
        imgs, target = data
        # 3
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            target = target.cuda()

        outputs = tudui(imgs)
        loss = loss_fn(outputs, target)

        # 优化器优化模型
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_step += 1

        # 减少打印次数
        if total_train_step % 100 == 0:
            print(f"训练次数{total_train_step},Loss is {loss.item()}")
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试开始
    tudui.eval()  # drop,batchnorm有作用
    total_test_loss = 0
    # 正确率
    total_accuracy = 0

    with torch.no_grad():
        for data in test_data_loader:
            imgs, target = data
            # 4
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                target = target.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, target)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy += accuracy.item()

    print(f"整体测试集的Loss:{total_test_loss}")
    print(f"整体测试集的正确率:{total_accuracy / test_data_size}")
    writer.add_scalar('test_loss', total_test_loss, total_train_step)
    writer.add_scalar('total_accuracy', total_accuracy, total_train_step)
    total_test_step += 1
    # 保存模型
    torch.save(tudui.state_dict(), f'./tudui{i}.pth')

writer.close()
