import numpy as np
from sklearn import metrics


def evulate(X, real_label, pre_label):
    print("轮廓系数为", metrics.silhouette_score(X, pre_label))
    print("DBI is ", metrics.davies_bouldin_score(X, pre_label))
    print("ARI is ", metrics.adjusted_rand_score(real_label, pre_label))
    print("FMI is ", metrics.fowlkes_mallows_score(real_label, pre_label))
    return metrics.silhouette_score(X, pre_label), metrics.davies_bouldin_score(X, pre_label), metrics.adjusted_rand_score(real_label, pre_label), metrics.fowlkes_mallows_score(real_label, pre_label)


def caculate_frequency(data: np.array, space):  # data是array类型
    """
    ## 计算数据集合的区间频数
    :param data:
    :param space: 区间大小
    :return:
    """
    frequency = []
    low, high = min(data), max(data)
    for i in np.linspace(low + space, high, -(low - high) // space):
        b = data[data < i]
        frequency.append(b[b > i - space].size)
    return frequency
