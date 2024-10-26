# 类的动态属性操作
# 方法一
class MyCls(object):
    def __init__(self):
        self._param = None
    def getParam(self):
        print(f"get param: {self._param}")
        return self._param
    def setParam(self, param):
        print(f"set param: {param}")
        self._param = param
    def delParam(self):
        print(f"del param: {self._param}")
        del self._param

    param = property(getParam, setParam, delParam)
    # 上这一行代码的作用是定义一个属性param，并绑定到getParam、setParam和delParam方法上。这意味着：
if __name__ == '__main__':
    cls = MyCls()
    cls.param = 10
    print(f"current is:{cls.getParam()}")
    del cls.param