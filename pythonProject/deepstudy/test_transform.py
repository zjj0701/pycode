from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "dataset/train/ant/0013035.jpg"
img = Image.open(img_path) #768x512

tensor_trans = transforms.ToTensor()
tensor_ = tensor_trans(img)
writer = SummaryWriter("logs")
writer.add_image("tensor_image",tensor_)
writer.close()
print(tensor_)