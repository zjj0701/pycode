# 梯度确认
from time import time

import numpy as np

from pythonProject.mtpro.my_first_cd.ch01.mnist import load_mnist
from pythonProject.mtpro.my_first_cd.test_fl_203 import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

'''
(fxh1 - fxh2) / (2*h)
'''
s = time()
grad_numerical = network.numerical_gradient(x_batch, t_batch)
d = time()
print(f"fp lasting:{d - s}")
'''
反向传播的起点是损失函数的导数值，初始化为1，表示从损失函数的输出开始向网络内部传播梯度。
'''
s = time()
grad_backprop = network.gradient(x_batch, t_batch)
d = time()
print(f"bp lasting:{d - s}")

for key in grad_numerical:
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(f"{key}:{diff}")
