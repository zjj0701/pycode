import time
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename="Dec_Test3.txt", filemode='a',
                    level=logging.DEBUG)


def get_num(func):
    logging.info("dect is running...")
    a: int = 2
    b: float = 10.5

    def wrapper(*args, **kwargs):
        logging.info("wrapper is running...")
        _sum = a + b  # 计算 a 和 b 的和
        # 确保 args 有足够的元素
        if len(args) >= 2:
            res = _sum + args[0] + args[1]  # 将第一个和第二个参数与 _sum 相加
        else:
            raise ValueError("Not enough arguments provided to the function")

        start = time.time()
        logging.info("appo wrapper is running...")
        # 传递 res 作为第一个参数，后面的 args[2:] 和 **kwargs 保持不变传递
        func(res, *args[1:], **kwargs)
        logging.info("after wrapper is running...")
        time.sleep(2)
        end = time.time()
        print(f"lasting: {end - start}")
        logging.info("wrapper is ending...")

    return wrapper


@get_num
def get_all(c: int, d: int, e: str):
    logging.info("main is starting")
    print(c, d, e)
    logging.info("main is ending")


# 调用 get_all 并传入 3 个参数
get_all(10, 20, "e_str")
