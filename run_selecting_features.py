"""run_selecting_features.py

selecting useful features for each fault
========================================

We determine whether a feature is useful by testing whether
the distribution of normal and abnormal invocations with
respect to it changes after the fault occurs.
分布是否变化

Feature candidate set
---------------------
In a microservice system, there are various metrics. In Train-Ticket data, we use
latency and HTTP status of each invocation, and CPU usage,
memory usage, network receive/send throughput, and disk
read/write throughput of each microservice as the features for
trace anomaly detection.
特征介绍
Note that we only consider the historical invocations of the same microservice
pair  to which this invocation belongs because the
underlying distributions with respect to the same feature can
vary vastly for different microservice pairs.


Parameters
  1. Input_file : the data after the fault happens (pkl)
    异常之后的pkl，即注入错误的
  2. History : all the historical  invocations of the same microservice pair
     1) in the last slot and 2) in the same slot of the last period (pkl)
     历史的
  3. output_file : the useful features of each invocation (dict)
  4. fisher_threshold : a given threshold to test whether the feature of the invocation is useful


"""

import pickle
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# from trainticket_config import FEATURE_NAMES
# 自定义所要使用的feature
FEATURE_NAMES = ['latency', 'http_status', 'cpu_use', 'mem_use_percent', 'mem_use_amount', 'file_write_rate',
                 'file_read_rate',
                 'net_send_rate', 'net_receive_rate']

DEBUG = False  # very slow


def distribution_criteria(empirical, reference, threshold):
    """This function has not been used
    """
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def fisher_criteria(empirical, reference, side='two-sided'):
    """This function has not been used
    """
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold) -> bool:
    """Testing whether the feature of the invocation is useful
    这块应该是参考α_after和 α_after计算，有参考历史

    :param empirical: the data after the fault happens 错误后数据(实验数据)
    :param reference: the normal data before the fault happens (contains twofold stage) 历史正常数据
    :param threshold: a given threshold to test whether the feature of the invocation is useful
    :return: bool type
    """
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    '''
    historical_mean * 0.01 + 0.01 对 均值的1% 添加了 0.01。这个值相当于对均值的一个小百分比的增量。
    使用NumPy的maximum函数，比较 historical_std 和上述计算的值，并选择它们中的较大值。
    这可以确保 historical_std 不会小于一个最小阈值，以防它太小。
    '''
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)

    '''
    1. `np.abs(reference - historical_mean)` 计算了 `reference` 数据集中每个数据点与历史均值 `historical_mean` 的绝对差异，即它们之间的距离。
    2. `np.mean(np.abs(reference - historical_mean))` 计算了这些绝对差异的平均值，表示了整个 `reference` 数据集的平均偏离程度。
    3. 最后，这个平均绝对差异值被除以历史标准差 `historical_std`，得到了 `ref_ratio`。这个值用来度量 `reference` 数据集相对于历史平均值的偏差
    如果 `ref_ratio` 较大，那么 `reference` 数据集与历史数据的差异较大，如果较小，则差异较小。
    '''
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio + 1.0


# @click.command('invocation feature selection')
# @click.option('-i', '--input', 'input_file', default="*.pkl", type=str)
# @click.option('-o', '--output', 'output_file', default='.', type=str)
# @click.option('-h', '--history', default='historical_data.pkl', type=str)
# @click.option("-f", "--fisher", "fisher_threshold", default=1, type=float)


def selecting_feature_main(input_file: str, output_file: str, history_file: str, fisher_threshold):
    """The main function to select the useful features

    :param input_file: the data after the fault happens (pkl) 输入错误后的pkl
    :param output_file: the useful features of each invocation (dict)  输出结果
    :param history_file: the normal data before the fault happens (contains twofold stage) 历史正常
    :param fisher_threshold: a given threshold to test whether the feature of the invocation is useful
    :return:
    """
    input_file_path = Path(input_file)
    logger.debug(f'{input_file}')

    output_file_path = Path(output_file)
    logger.debug(f'{output_file}')

    history_file_path = Path(history_file)
    # logger.debug('history file ---' + history_file)
    with open(history_file_path, 'rb') as f:
        history_df = pickle.load(f)
    # logger.debug('print history.info()')
    # print(history.info())

    with open(input_file, 'rb') as f:
        input_df = pickle.load(f)

    # input_df = pd.DataFrame(input_df)
    # logger.debug('input_file' + input_file)
    # print(input_df.info())

    '''
    set_index(keys=['source', 'target'], drop=True) 设置了DataFrame的索引。keys参数指定了要用作索引的列，这里是'source'和'target'列。
    drop=True 表示将原来的列从DataFrame中删除，只保留这两列作为索引。
    sort_index() 对DataFrame按照索引进行排序。这将确保DataFrame按照指定的索引列的值的顺序进行排序。
    '''
    input_df = input_df.set_index(keys=['source', 'target'], drop=True).sort_index()
    input_df['http_status'] = pd.to_numeric(input_df['http_status'])

    history_df = history_df.set_index(keys=['source', 'target'], drop=True).sort_index()
    history_df['http_status'] = pd.to_numeric(history_df['http_status'])

    logger.info('df 加载完成')

    '''
    np.unique(df.index.values) 和 np.uniq1ue(history.index.values)：
    首先，它使用np.unique()函数从两个不同的索引数组中提取唯一的值。
    df.index.values表示从DataFrame df 的索引中提取唯一的值，history.index.values表示从名为history的对象的索引中提取唯一的值。
    np.intersect1d()：然后，使用np.intersect1d()函数来找到这两个唯一值数组的交集。
    这意味着indices将包含同时出现在df的索引和history的索引中的唯一值
    indices是输入和历史中都有的source-target对
    '''
    indices = np.intersect1d(np.unique(input_df.index.values), np.unique(history_df.index.values))
    logger.info('获取到所有 source-target pair')


    '''
    indices 形如
    ('istio-ingressgateway', 'ts-assurance-service')
    ('istio-ingressgateway', 'ts-auth-service')
    ('istio-ingressgateway', 'ts-cancel-service')
    ('istio-ingressgateway', 'ts-consign-service')
    。。。
    '''
    # logger.debug(indices)

    '''
    创建了一个名为 useful_features_dict 的字典（dictionary），其中每个键对应的值是一个空列表。
    这看起来是使用Python标准库 collections 模块中的 defaultdict 类来创建的。
    '''
    useful_features_dict = defaultdict(list)

    if DEBUG:
        plot_dir = output_file_path.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)

    # 遍历所有可能的(source-target) 的所有feature
    for (source, target), feature in tqdm(product(indices, FEATURE_NAMES)):  # 笛卡尔积

        # 这个服务pair的这个feature的值，进行排序。一个一维数组。实验数据也就是异常后数据
        empirical = np.sort(input_df.loc[(source, target), feature].values)

        reference = np.sort(history_df.loc[(source, target), feature].values)
        # p_value = ks_2samp(
        #     empirical, reference, alternative=ALTERNATIVE[feature]
        # )[1]
        p_value = -1
        fisher = stderr_criteria(empirical, reference, fisher_threshold)
        # fisher = distribution_criteria(empirical, reference,fisher_threshold)
        # if target == 'ts-station-service':
        #    print(source,feature,fisher)
        # fisher = fisher_criteria(empirical, reference, side=ALTERNATIVE[feature])
        # if target == 'ts-food-service':
        #     logger.debug(f"{source} {target} {feature} {fisher} "
        #                  f"{np.mean(empirical)} {np.mean(reference)} {np.std(reference)}")
        if fisher:
            useful_features_dict[(source, target)].append(feature)
        try:
            if DEBUG:
                fig = Figure(figsize=(4, 3))
                # x = np.sort(np.concatenate([empirical, reference]))
                # print('DEBUG:')
                # print(empirical,reference)
                plt.clf()
                sns.distplot(empirical, label='Empirical')
                sns.distplot(reference, label='Reference')
                plt.xlabel(feature)
                plt.ylabel('PDF')
                plt.legend()
                plt.title(f"{source}->{target}, ks={p_value:.2f}, fisher={fisher:.2f}")
                plt.savefig(
                    plot_dir / f"{input_file_path.name.split('.')[0]}_{source}_{target}_{feature}.pdf",
                    bbox_inches='tight', pad_inches=0
                )

        except Exception as e:
            print(f'{e}')
        # logger.debug(f"{input_file_path.name} {source} {target} {feature} {fisher}")
        # useful_features_dict[(source, target)].append(feature)

    # logger.debug(f"{input_file.name} {dict(useful_features_dict)}")
    # with open(output_file, 'w+') as f:
    #     print(dict(useful_features_dict), file=f)

    with open(Path(output_file), 'wb') as f:
        pickle.dump(useful_features_dict, f)


if __name__ == '__main__':
    # input_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\admin-order_abort_1011_data.pkl'
    # history = r'E:\AIOPs\TraceRCA-main\A\uninjection\pkl_3_data.pkl'
    # output_file = r'E:\AIOPs\TraceRCA-main\A\uninjection\useful_feature_2'

    # input file也需要先经过process
    input_file = Path(r'./A/exception/admin-order_abort_1011_processed.pkl')
    input_file_name = input_file.stem
    history_file = Path(r'./A/uninjection/pkl_3_processed.pkl')
    history_name = history_file.stem

    # output_file = r'./A/uninjection/useful_feature_2'

    output_feature_base = Path('./A/output/feature')
    output_file = output_feature_base / ("input--" + input_file_name + "--history--" + history_name + ".pkl")
    fisher_threshold = 1
    logger.info('开始执行select main方法')

    selecting_feature_main(input_file=str(input_file), output_file=str(output_file), history_file=str(history_file),
                           fisher_threshold=fisher_threshold)
