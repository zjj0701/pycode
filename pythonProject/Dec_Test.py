import time
import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def print_time(func):
    print("decorator is running...")
    logging.info("decorator is running...")

    def wrapper(*args, **kwargs):
        logging.info("wrapper is running...")
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(end - start)
        logging.info("wrapper is ending...")

    return wrapper


@print_time
def print_hello(name):
    logging.info("main func is beginning")
    print("hello", name)
    logging.info("main func is ending")


print(print_hello("James"))
