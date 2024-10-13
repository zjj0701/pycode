import numpy as np


# 计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# lr:学习率,step_num:重复次数
def gradient_descent(f, int_x, lr=0.01, step_num=100):
    x = int_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    # 此时的学习率是0.1
    res = gradient_descent(function_2, init_x, lr=0.1, step_num=100)
    print(f"学习率是0.1：{res}")
    # 学习率过大为10.0
    res = gradient_descent(function_2, init_x, lr=10.0, step_num=100)
    print(f"学习率是10：{res}")
    # 学习率过小为1e-10
    res = gradient_descent(function_2, init_x, lr=1e-10, step_num=100)
    print(f"学习率是1e-10：{res}")
