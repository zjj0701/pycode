class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1
