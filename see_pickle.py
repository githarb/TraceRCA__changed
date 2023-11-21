import pickle
from pathlib import Path
import pandas as pd

pd.set_option('display.max_columns', None)

import pandas as pd
from loguru import logger
from tqdm import tqdm

'''
2.pkl 里面的特征只有延迟，http? 也不是，只是有些记录是这样
admin-order 里面的特征有cpu，mem，net等等
'''


def main():
    # file = './A/uninjection/2.pkl'
    # pkl = loadPickle(file)
    # print(len(pkl))
    #
    # df1 = pd.DataFrame(pkl)
    # df2 = pd.DataFrame()
    # # multi = [i for i in pkl if len(i['s_t']) > 4]
    # # print(multi[2])
    #
    # for num in tqdm(range(len(pkl)), desc="Processing data"):
    #     # for num in range(len(pkl_data)):
    #     df_new = pd.DataFrame(pkl[num])
    #     df2 = pd.concat([df2, df_new], ignore_index=True)
    #
    # if df1.equals(df2):
    #     print('完全相同')
    # else:
    #     print('不完全相同')
    #
    # path1 = Path('./unrelated/pkl2_df1.pkl')
    # path2 = Path('./unrelated/pkl2_df2.pkl')
    # with open(path1, 'wb') as f:
    #     pickle.dump(df1, f)
    #
    # with open(path2, 'wb') as f:
    #     pickle.dump(df2, f)

    df = loadPickle('./A/exception/admin-order_abort_1011_processed.pkl')
    print(df.head())


def loadPickle(path):
    file = Path(path)
    with open(file, 'rb') as f:
        pkl = pickle.load(f)
    return pkl


def compareConcatenateAndOneTime():
    pass


if __name__ == '__main__':
    main()
