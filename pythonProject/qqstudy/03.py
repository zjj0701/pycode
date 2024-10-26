# 类的动态属性操作
# 方法2:常用

class MyCls(object):
    def __init__(self):
        self._param = None

    @property
    def param(self):
        print(f"get param: {self._param}")
        return self._param

    @param.setter
    def param(self, param):
        print(f"set param: {param}")
        self._param = param

    @param.deleter
    def param(self):
        print(f"del param: {self._param}")
        del self._param

if __name__ == '__main__':
    cls = MyCls()
    cls.param = 10
    print(f"current is:{cls.param}")
    del cls.param