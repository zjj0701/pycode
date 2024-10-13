import numpy as np


# 交叉熵
def cross_entropy_error(y, t):
    """
    当出现np.log(0)时，np.log(0)会变为负无限大的-inf，这样一来就会导致后续计算无法进行。
    作为保护性对策，添加一个微小值可以防止负无限大的发生
    :param y:
    :param t:
    :return:
    """
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
