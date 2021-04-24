import random

import numpy as np


def create_multi_normal_datasets(dataFile, lableFile):
    """
    ## 随机产生服从高斯分布的三个二维数据集——组成一个混合高斯分布
    :param dataFile: 数据集保存地址
    :param lableFile: 数据标签保存地址
    :return: None
    """

    mean1 = [-1, 2]
    cov1 = [[0.774, -0.268], [-0.268, 0.258]]
    data1 = np.random.multivariate_normal(mean1, cov1, 200)
    label1 = np.array([0 for _ in range(200)])

    mean2 = [3, 1]
    cov2 = [[0.651, -0.294], [-0.294, 0.197]]
    data2 = np.random.multivariate_normal(mean2, cov2, 400)
    label2 = np.append(label1, np.array([1 for _ in range(400)]), 0)

    mean3 = [2, 5]
    cov3 = [[0.912, 0.323], [0.323, 0.265]]
    data3 = np.random.multivariate_normal(mean3, cov3, 400)
    label3 = np.append(label2, np.array([2 for _ in range(400)]), 0)

    np.savez(dataFile, data1=data1, data2=data2, data3=data3)
    np.savez(lableFile, label=label3)
    return


# sigma = 0.05; r = 0.01; k = 3000; t_max = 300.; S0 = 10.
# T = np.arange(0, 300, 0.1)
# T1 = np.arange(300, 600, 0.1)
# y1=gbm(sigma, r, k, t_max, S0, I=1)
def create_one_brown_datasets(sigma, r, k, t_max, S0, I=1):
    """
    ## 产生一维布朗运动模拟数值
    :param sigma:
    :param r:
    :param k:
    :param t_max:
    :param S0:
    :param I:
    :return:
    """
    phi = np.random.normal(size=(k, I))
    dt = t_max / k
    y = sigma * np.sqrt(dt) * phi + (r - sigma ** 2 / 2.) * dt
    return S0 * np.exp(np.cumsum(y, axis=0))


def create_one_normal_datasets(mu, sigma, alpha, k):
    '''
    ## 这里通过服从高斯分布的随机函数来伪造数据集
    :param mu0: 高斯0的均值
    :param sigma0: 高斯0的方差
    :param mu1: 高斯1的均值
    :param sigma1: 高斯1的方差
    :param alpha0: 高斯0的系数
    :param alpha1: 高斯1的系数
    :return: 混合了两个高斯分布的数据
    '''
    # 定义数据集长度为3000
    length = 3000
    dataSet = []
    for i in range(k):
        d = np.random.normal(mu[i], sigma[i], int(length * alpha[i]))
        dataSet.extend(d)
    # 对总的数据集进行打乱（其实不打乱也没事，只不过打乱一下直观上让人感觉已经混合了
    # 读者可以将下面这句话屏蔽以后看看效果是否有差别）
    random.shuffle(dataSet)
    return dataSet


def create_one_uniform_datasets(low, high, size):
    """
    ## 随机产生一维均匀分布数据集
    :param low: 均匀分布最小值
    :param high: 均匀分布最大值
    :param size: 数据集大小
    :return:
    """
    return np.random.uniform(low, high, size)
