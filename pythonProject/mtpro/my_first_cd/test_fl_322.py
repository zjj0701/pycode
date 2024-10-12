import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1.0, 1.0, 2.0])
print(x)
y = x > 0  # 进行bool运算 dtype = bool [False  True  True]
print(y)
# 转换为int
print(y.astype(int))  # [0 1 1]


def step_function(x):
    """
    阶跃函数
    :param x: 
    :return: 
    """""
    return np.array(x > 0, dtype=np.int_)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # np.exp(-x)=exp(-x)


def relu(x):
    return np.maximum(0, x)


# x = np.arange(-5.0, 5.0, 0.1)  # numpy的range函数
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # 指定yxais范围
# plt.show()

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定yxais范围
plt.show()
