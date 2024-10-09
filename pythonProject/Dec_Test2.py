import time
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def get_num(func):
    logging.info("dect is running...")
    a: int = 2
    b: float = 10.5

    def wrapper(*args, **kwargs):
        logging.info("wrapper is running...")
        res = a + b + args[0] + args[1]
        start = time.time()
        logging.info("appo wrapper is running...")
        func(res, *args[1:], **kwargs)
        logging.info("after wrapper is running...")
        time.sleep(2)
        end = time.time()
        print(f"lasting:{end - start}")
        logging.info("wrapper is ending...")

    return wrapper


@get_num
def get_all(c: int, d: int, e: str):
    logging.info("main is starting")
    print(c, d)
    logging.info("main is ending")


get_all(10, 20, "last parameter")
