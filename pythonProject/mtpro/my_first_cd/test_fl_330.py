# 多维数组
import numpy as np

A = np.array([1, 2, 3, 4])
print(A)  # [1 2 3 4]
print(A.ndim)  # 1 数组的维数
print(A.shape)  # (4,) 数组的形状
print(A.shape[0])  # 4

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)  #
print(B.ndim)  # 2 数组的维数
print(B.shape)  # (3,2) 数组的形状 3×2
print(B.shape[0])  # 3

C = np.array([7, 8])  # 当B的维数是一维的时候，此时也成立 2 (2x3) = 3 注意看元素的个数是否一致
print(np.dot(B, C))  # 乘法
