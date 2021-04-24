import time

import numpy as np

from util.caculate import evulate
from util.pro_density_func import prob


def gaussianCluster_MEEM(traindataMat, testdataMat, testLabelMat, K, maxIter, delta=2.8):
    MEEMLabel = []
    FMI = 0
    label = list(set(testLabelMat))
    # traindataMat是已标注的数据集(x,y)且为Mat类型,y为最后一列
    # m为样本数量,n为特征数
    traindataSize = traindataMat.shape[0]  # 已标定样本数量
    testdataSize, n = testdataMat.shape  # 未标定样本数量

    # 1.初始化各高斯混合成分参数
    gamma = 0.2
    # 较小的[0,1]之间的
    mu = [0 for _ in range(K)]
    sigma = [0 for _ in range(K)]
    alpha = [0 for _ in range(K)]
    S = []  # 记录每类已经标记的样本数量
    traindataK = []  # 记录每类已经标定的样本
    for i in range(K):
        traindataMatK = traindataMat[np.where(traindataMat[:, -1] == label[i])[0], :].T[:-1].T  # 获取第i类中已经标定第样本
        num, feature = traindataMatK.shape  # num为第i类的已标记的样本数
        S.append(num)
        traindataK.append(traindataMatK)

        # 1.1计算第i类中已标定的样本的x均值作为该类别mu的初始值
        mu[i] = np.array(traindataMatK.mean(axis=0))[0]

        # 1.2计算第i类中已标定的样本的协方差作为该类别协方差矩阵的初始值
        for j in range(num):
            sigma[i] += (traindataMatK[j, :] - mu[i]).T * (traindataMatK[j, :] - mu[i])
        sigma[i] /= num

        # 1.3 n_k/n 作为混合系数alpha的初始值
        alpha[i] = num / traindataSize

    ## MEEM算法求解
    z = np.mat(np.zeros((testdataSize, K)))  # 测试集隐变量后验概率z

    zz = np.mat(np.zeros((testdataSize, K)))
    startTime = time.time()
    Q = 0
    curMEEM = 0
    for i in range(maxIter):

        # 2.计算测试集隐变量后验概率z
        totals = []
        for j in range(testdataSize):
            sumAlphaMulP = 0
            total = 0
            for k in range(K):
                z[j, k] = prob(testdataMat[j, :], mu[k], sigma[k]) ** gamma  # 与普通EM算法的区别之处
                zz[j, k] = alpha[k] * prob(testdataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += z[j, k]
                total += zz[j, k]
            totals.append(total)
            for k in range(K):
                z[j, k] /= sumAlphaMulP
        sumz = np.sum(z, axis=0)

        # 3.更新模型参数
        for k in range(K):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = np.mat(np.zeros((n, n)))
            # 3.1更新均值mu
            mu[k] += traindataK[k].sum(axis=0)
            for j in range(testdataSize):
                mu[k] += z[j, k] * testdataMat[j, :]
            mu[k] /= (sumz[0, k] + S[k])
            # 3.2更新协方差矩阵sigma
            # 3.2.1已标注样本部分
            for j in range(S[k]):
                sigma[k] += (traindataK[k][j, :] - mu[k]).T * (traindataK[k][j, :] - mu[k])
            # 3.2.2未标注样本部分
            for j in range(testdataSize):
                sigma[k] += z[j, k] * (testdataMat[j, :] - mu[k]).T * (testdataMat[j, :] - mu[k])
            sigma[k] /= (sumz[0, k] + S[k])
            # 3.3.3更新混合系数alpha
            alpha[k] = (sumz[0, k] + S[k]) / (traindataSize + testdataSize)

        # 4.对未标定样本进行标定
        m, n = np.shape(testdataMat)
        clusterAssign = np.mat(np.zeros((m, 2)))
        for mm in range(m):
            # amx返回矩阵最大值，argmax返回矩阵最大值所在下标
            clusterAssign[mm, :] = np.argmax(z[mm, :]), np.amax(z[mm, :])
        stopTime = time.time()
        pre_label = [int(c) for c in clusterAssign[:, 0]]

        label = evulate(testdataMat, testLabelMat, pre_label)[3]
        MEEMLabel.append(label)
        FMI += label
        Q_new = np.sum(np.log(np.array(totals)))  # 对数似然函数值
        # 5.2收敛条件2:gamma不断趋向1
        if abs(Q_new - Q) < 1:
            continue
        else:
            Q = Q_new
        if gamma < 1:
            gamma = min(delta * gamma, 1)
        else:
            continue
    return pre_label, MEEMLabel, FMI / maxIter
