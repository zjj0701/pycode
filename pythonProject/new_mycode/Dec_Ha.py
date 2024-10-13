import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    filename="Dec_Ha.txt", filemode='a',
                    level=logging.DEBUG)


def H_n(n, a, b, c):
    logging.info("begin...")
    if n == 1:
        return
    H_n(n - 1, a, b, c)
    H_n(1, a, c, b)
    H_n(n - 1, b, c, a)

H_n(3,'a','b','c')
