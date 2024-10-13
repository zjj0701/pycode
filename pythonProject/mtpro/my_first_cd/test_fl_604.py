import numpy as np

from pythonProject.mtpro.my_first_cd.commons.functions import softmax, cross_entropy_error


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 高斯分布初始化

    def predict(self, X):
        return np.dot(X, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    net = simpleNet()
    print(f"net's w is:{net.W}")
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(f"predict is:{p}")
    print(f"the max index of p is:{np.argmax(p)}")

    t = np.array([0, 0, 1])
    print(f"the loss is:{net.loss(x, t)}")
