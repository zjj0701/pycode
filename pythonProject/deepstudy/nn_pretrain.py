import torchvision

from torch import nn

# vgg16 前置的网络结构,提取特殊的特征K
vgg_16_false = torchvision.models.vgg16(pretrained=False)
vgg_16_true = torchvision.models.vgg16(pretrained=True)
print(f"before:{vgg_16_true}")
# 怎么加一层？
vgg_16_true.add_module("add_Linear", nn.Linear(1000, 10))
print(f"after:{vgg_16_true}")
# 怎么在classfier里面加或者修改？
vgg_16_false.classifier[7] = nn.Linear(1000,10)
print(f"classifier after :{vgg_16_true}")