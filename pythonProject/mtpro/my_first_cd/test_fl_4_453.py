import numpy as np

from pythonProject.mtpro.my_first_cd.ch01.mnist import load_mnist
from pythonProject.mtpro.my_first_cd.test_fl_4_451 import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size // batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
'''
对训练数据和测试记录识别精度
'''
for i in range(iters_num):
    # 获取mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % iter_per_epoch == 0:
        tran_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(tran_acc)
        test_acc_list.append(test_acc)
        print(f"train acc:{tran_acc}, test acc :{test_acc}")
