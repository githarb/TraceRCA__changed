import pickle
from pathlib import Path

import pandas as pd

from tqdm import tqdm

# file = r'E:\AIOPs\TraceRCA-main\A\uninjection\3.pkl' #下载下来的原始文件

file = Path('./A/exception/admin-order_abort_1011.pkl')
with open(str(file), 'rb') as f:
    pkl_data = pickle.load(f)

print('pickle 加载完成')
print(pkl_data[1])


df_list = []
for num in tqdm(range(len(pkl_data)), desc="拼接df"):
    # for num in range(len(pkl_data)):
    df_new = pd.DataFrame(pkl_data[num])
    df_list.append(df_new)

df = pd.concat(df_list, ignore_index=True)

'''
必须要一行一行拼接，不能直接在初始化里传入，两个不一样
'''
print('df 拼接完成')


source = []
target = []
for num in tqdm(range(len(df)), desc="处理source target"):
# for num in range(len(df)):
    source_new, target_new = df.loc[num, 's_t']
    source.append(source_new)
    target.append(target_new)
df['source'] = source
df['target'] = target



# df.to_csv(r'F:\ruijie\AIOPS\TraceRCA-main\A\uninjection\pkl_2_data.csv')

# def save_dict(data, name):
#     with open(name, 'wb') as f:
#         pickle.dump(data, f)

# 获取文件名，不带扩展名
file_name = file.stem

output_file = file.parent / (file_name + '_processed.pkl')


with open(output_file, 'wb') as f:
    pickle.dump(df, f)

# original
# save_dict(df,r'F:\AIOPS\TraceRCA-main\A\uninjection\pkl_3_data.pkl') #重新调整后符合代码输入格式的数据
# save_dict(df, r'./A/uninjection/pkl_3_processed.pkl')

print('处理完成')
# '''反向解析时间戳'''
# import time
# def stampToTime(stamp):
#     datatime = time.strftime('%T-%m-%d %H:%M:%S',time.localtime(float(str(int(stamp))[0:10])))
#     datatime = datatime + '.' + str(stamp)[10:]
#     return datatime
#
