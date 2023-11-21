import pickle
from pathlib import Path

from loguru import logger

from see_pickle import loadPickle


def main():
    dict = loadPickle('./A/output/feature/input--admin-order_abort_1011_processed--history--pkl_3_processed.pkl')
    for item in dict:
        logger.info(item)
        l = dict[item]
        logger.info(l)

if __name__ == '__main__':
    main()
