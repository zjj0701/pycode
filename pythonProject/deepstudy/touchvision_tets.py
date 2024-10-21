import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=True, transform=dataset_transform,
                                         download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=False, transform=dataset_transform,
                                        download=True)

print(test_set[0])

writer = SummaryWriter("./logs/p10")
for i in range(10):
    img, target = train_set[i]
    writer.add_image('train/{}'.format(i), img, i)
writer.close()
