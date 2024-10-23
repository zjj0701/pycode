import torch
import torchvision.models

vgg_16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1:保存模型以及参数
torch.save(vgg_16, "my_vgg16.pth")
# 只保存参数，不保存模型(官方推荐)
torch.save(vgg_16.state_dict(), "my_vgg16_no.pth")
