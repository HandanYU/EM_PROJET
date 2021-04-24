# 高斯分布的概率密度函数
import time

import numpy as np
from numpy import *

from util.caculate import evulate
from util.pro_density_func import prob


# EM算法
def gaussianCluster(traindataMat, testdataMat, testdataLabelMat, K, maxIter):
    MEMLabel = []
    # traindataMat是已标注的数据集(x,y)且为Mat类型,y为最后一列
    label = list(set(testdataLabelMat))
    traindataSize = traindataMat.shape[0]  # 已标定样本数量
    testdataSize, n = testdataMat.shape  # 未标定样本数量

    # 1.初始化各高斯混合成分参数
    mu = [0 for _ in range(K)]
    sigma = [0 for _ in range(K)]
    alpha = [0 for _ in range(K)]
    S = []  # 记录每个类别已标定样本数量
    traindataK = []  # 记录每个类别已经标定样本数量
    pre_label = []
    for i in range(K):
        traindataMatK = traindataMat[np.where(traindataMat[:, -1] == label[i])[0], :].T[:-1].T  # 获取第i类中已经标定第样本()

        traindataK.append(traindataMatK)
        num, feature = traindataMatK.shape  # num为第i类的已标记的样本数
        S.append(num)
        # 1.1计算第i类中已标定的样本的x均值作为该类别mu的初始值
        mu[i] = traindataMatK.mean(axis=0)
        mu[i] = [s / num for s in traindataMatK.sum(axis=0).tolist()[0]]

        # 1.2计算第i类中已标定的样本的协方差作为该类别协方差矩阵的初始值
        for j in range(num):
            sigma[i] += (traindataMatK[j, :] - mu[i]).T * (traindataMatK[j, :] - mu[i])
        sigma[i] /= num

        # 1.3 n_k/n 作为混合系数alpha的初始值
        alpha[i] = num / traindataSize

    gamma = np.mat(np.zeros((testdataSize, K)))
    trainXSum = traindataMat[:, :-1].sum(axis=0)
    startTime = time.time()
    Q = 0
    ## EM算法求解模型参数
    for i in range(maxIter):
        # 2.计算测试集隐变量后验概率
        totals = []
        for j in range(testdataSize):
            sumAlphaMulP = 0

            for k in range(K):
                gamma[j, k] = alpha[k] * prob(testdataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += gamma[j, k]
            totals.append(sumAlphaMulP)
            for k in range(K):
                gamma[j, k] /= sumAlphaMulP
        sumGamma = np.sum(gamma, axis=0)

        # 3.更新模型参数mu,sigma,alpha
        for k in range(K):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = np.mat(np.zeros((n, n)))
            mu[k] += traindataK[k].sum(axis=0)

            #  3.1.计算新均值向量
            for j in range(testdataSize):
                mu[k] += gamma[j, k] * testdataMat[j, :]
            mu[k] /= (sumGamma[0, k] + S[k])

            # 3.2. 计算新的协方差矩阵
            # 3.2.1 已标定样本集部分
            for j in range(S[k]):
                sigma[k] += (traindataK[k][j, :] - mu[k]).T * (traindataK[k][j, :] - mu[k])
                # 3.2.1 未标定样本集部分
            for j in range(testdataSize):
                sigma[k] += gamma[j, k] * (testdataMat[j, :] - mu[k]).T * (testdataMat[j, :] - mu[k])
            sigma[k] /= (sumGamma[0, k] + S[k])

            # 3.3. 计算新混合系数
            alpha[k] = (sumGamma[0, k] + S[k]) / (traindataSize + testdataSize)

            ## 对未标记样本进行标记
        m, n = np.shape(testdataMat)
        clusterAssign = np.mat(np.zeros((m, 2)))
        for mm in range(m):
            # amx返回矩阵最大值，argmax返回矩阵最大值所在下标
            clusterAssign[mm, :] = np.argmax(gamma[mm, :]), np.amax(gamma[mm, :])  # 15.确定x的簇标记lambda
        stopTime = time.time()

        pre_label = [label[int(c)] for c in clusterAssign[:, 0]]  # 获取预测标记结果

        Q_new = np.sum(np.log(np.array(totals)))
        if abs(Q_new - Q) < 1:
            continue
        else:
            Q = Q_new
        mem = evulate(testdataMat, testdataLabelMat, pre_label)[3]  # 评价模型
        MEMLabel.append(mem)
    return pre_label, MEMLabel
