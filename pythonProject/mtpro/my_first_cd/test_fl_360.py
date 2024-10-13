import sys, os
import numpy as np
from PIL import Image

sys.path.append(os.pardir)
from ch01.mnist import load_mnist


def img_show(img):
    # 转为PIL用的数据对象
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# flatten=True:图像以一维保存，显示需要展开为28*28，reshape指定形状
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
img = x_train[0]
label = t_train[0]
print(f"label={label}")
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)
