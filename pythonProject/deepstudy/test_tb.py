from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = 'dataset/train/ant/5650366_e22b7e1065.jpg'
img = Image.open(img_path)
img_arr = np.array(img)
print(img_arr.shape)  # 形状是(512, 768, 3) H W　C　通道在最后
writer.add_image("test", img_arr, 1, dataformats="HWC")
for i in range(100):
    writer.add_scalar("y = x", i, i)
writer.close()
