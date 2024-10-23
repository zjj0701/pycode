import torch
import torchvision

moudel = torch.load("my_vgg16.pth")
print(moudel)

moudel = torch.load("my_vgg16_no.pth")
print(moudel)

# 怎么恢复到第一种？
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("my_vgg16_no.pth"))
print(vgg16)
