import numpy as np


def prob(x, mu, sigma):
    """
    # 计算高斯混合模型的概率密度函数，当低维时候，也就是矩阵可逆
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    n = x.shape[1]
    expOn = float(-0.5 * (x - mu) * (sigma.I) * (x - mu).T)
    divBy = pow(2 * np.pi, n / 2) * pow(np.linalg.det(sigma), 0.5)
    return pow(np.e, expOn) / divBy


def prob_plus(x, mu, sigma):
    n = np.shape(x)[1]
    """
    按照给定的mu和covariance生成数据集X的概率分布
    """
    # 加一个微小的单位矩阵应对矩阵不可逆的情况
    covdet = np.linalg.det(sigma + np.eye(n) * 1e-3)  # 协方差矩阵的行列式
    covinv = np.linalg.inv(sigma + np.eye(n) * 1e-3)  # 协方差矩阵的逆

    xdiff = x - mu  # (m,n)的矩阵
    expOn = float(-0.5 * xdiff * covinv * (xdiff.T))
    divBy = pow(2 * np.pi, n / 2) * pow(covdet, 0.5)

    prob = pow(np.e, expOn) / divBy
    return prob
