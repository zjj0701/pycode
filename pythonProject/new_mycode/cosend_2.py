import logging
from inspect import getgeneratorstate

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def simple_(a):
    logging.debug('begin...  a value is %s',a)
    b = yield a
    logging.debug('b value is :%s', b)
    c = yield a + b
    logging.debug('c value is :%s', c)


if __name__ == '__main__':
    my = simple_(14)
    print(getgeneratorstate(my))
    next(my)
    print(getgeneratorstate(my))
    my.send(28)
    my.send(29)
    my.send(99)
    print(getgeneratorstate(my))
