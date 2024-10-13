import logging

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def simple_():
    logging.debug('begin...')
    x = yield
    logging.debug('receive value:')


if __name__ == '__main__':
    my = simple_()
    my.send(1)
