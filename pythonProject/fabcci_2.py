import time
from sys import meta_path


def getnum(n: int) -> int:
    if n in maps:
        return maps[n]
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        res =  getnum(n - 1) + getnum(n - 2)
    maps[n] = res
    return res

first = 1
second = 1
maps = dict()
start = time.time()
print(getnum(40))
end = time.time()
print(f'time is:{end - start}s')
