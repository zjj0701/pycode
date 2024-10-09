def decator1(func):
    def wrapper():
        print("装饰器1执行")
        func()

    return wrapper


def decator2(func):
    def wrapper():
        print("装饰器2执行")
        func()

    return wrapper


@decator1
@decator2
def hello_dec():
    print("hello,world")


hello_dec()
