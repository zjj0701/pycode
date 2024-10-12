import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
# 元素分别相乘
print(w * x)
# sum每一个相加求和
print(np.sum(w * x) + b)


# 感知机--多层感知机

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
