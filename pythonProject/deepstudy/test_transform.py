from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/bees/205835650_e6f2614bee.jpg"
img = Image.open(img_path)  # 768x512

tensor_trans = transforms.ToTensor()
tensor_ = tensor_trans(img)
writer = SummaryWriter("logs")

# normalize
print(tensor_[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_)
print(img_norm[0][0][0])

# resize
print(img.size)
trans_size = transforms.Resize((512, 512))
img_size = trans_size(img)
print(img_size)

writer.add_image("tensor_image", tensor_)

writer.close()
