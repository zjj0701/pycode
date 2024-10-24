import torch
import torchvision.transforms
from PIL import Image
from torch import nn

img_path = "./a.png"
img = Image.open(img_path)
img = img.convert("RGB")
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()
     ]
)
img = transform(img)
print(img)


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


model = torch.load("tudui10.pth",map_location=torch.device('cpu'))
img = torch.reshape(img, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))

