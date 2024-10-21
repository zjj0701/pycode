import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

test_data = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=False, transform=dataset_transform,
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

writer = SummaryWriter("./dataset/logs/")
step = 0
for data in test_loader:
    img, label = data
    # print(f"{img.shape},{label}")
    writer.add_image("imgi", img, step)
    step +=1

writer.close()