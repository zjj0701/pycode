import time


def getnum(n:int)->int:
    if n == 1:
        return 1
    elif n == 2:
        return 1
    else:
        return getnum(n-1)+getnum(n-2)
first = 1
second = 1
start = time.time()
print(getnum(40))
end = time.time()
print(f'time is:{end -start}s')