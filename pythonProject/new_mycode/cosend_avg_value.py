import time
import logging
from inspect import getgeneratorstate

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def avgrager():
    total: float = 0.0
    count: int = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total / count

if __name__ == '__main__':
    co = avgrager()
    next(co)
    print(co.send(10))
    print(co.send(30))
    co.close()
    print(co.send(5))