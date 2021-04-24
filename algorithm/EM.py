import time

import numpy as np
from numpy import *
from sklearn import metrics
from sklearn.cluster import KMeans

from util.caculate import evulate
from util.pro_density_func import prob


# EM算法

def tranditionalGaussianCluster(testdataMat, labelMat, maxIter, K, mu, density_func=prob):
    EMLabel = []
    start_time = time.time()
    # EM_FMI=[]
    # print(mu)
    m, n = testdataMat.shape
    # print(n)
    # 初始化各高斯混合成分参数
    alpha = [1 / K] * K
    sigma = [mat(np.identity(testdataMat.shape[1], dtype=np.float64)) for x in range(K)]
    gamma = mat(zeros((m, K)))
    pre_label = []
    Q = 0
    for i in range(maxIter):
        print("iter:", i + 1)
        totals = []
        for j in range(m):
            sumAlphaMulP = 0
            for k in range(K):
                gamma[j, k] = alpha[k] * density_func(testdataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += gamma[j, k]
            totals.append(sumAlphaMulP)
            for k in range(K):
                gamma[j, k] /= sumAlphaMulP
        sumGamma = sum(gamma, axis=0)
        for k in range(K):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = mat(zeros((n, n)))
            for j in range(m):
                mu[k] += gamma[j, k] * testdataMat[j, :]
            mu[k] /= sumGamma[0, k]
            for j in range(m):
                sigma[k] += gamma[j, k] * (testdataMat[j, :] - mu[k]).T * (testdataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]
            alpha[k] = sumGamma[0, k] / m
        ## step 1: init centroids
        clusterAssign = np.mat(np.zeros((m, 2)))
        for mm in range(m):
            # amx返回矩阵最大值，argmax返回矩阵最大值所在下标
            clusterAssign[mm, :] = np.argmax(gamma[mm, :]), np.amax(gamma[mm, :])  # 15.确定x的簇标记lambda

        pre_label = [int(c) for c in clusterAssign[:, 0]]
        # print(metrics.fowlkes_mallows_score(labelMat,pre_label))
        # EMLabel.append(metrics.silhouette_score(testdataMat,pre_label))
        # EM_FMI.append(metrics.fowlkes_mallows_score(labelMat,pre_label))
        EMLabel.append(metrics.fowlkes_mallows_score(labelMat, pre_label))
        Q_new = np.sum(np.log(np.array(totals)))
        evulate(testdataMat, labelMat, pre_label)
        if abs(Q_new - Q) < 1:
            # stop_time=time.time()
            continue
        else:
            Q = Q_new
    return pre_label, EMLabel  # ,stop_time-start_time


## 需要测试集的EM算法，即利用K-Means进行初始化的EM算法
def initCent(traindataMat, K):
    estimator = KMeans(n_clusters=K)  # 构造聚类器
    estimator.fit(traindataMat)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心
    return np.array(centroids)


def KMeans_EM(traindataMat, testdataMat, labelMat, maxIter, K):
    EMLabel = []
    mu = np.mat(initCent(traindataMat, K))
    m, n = shape(testdataMat)
    # 初始化各高斯混合成分参数
    alpha = [1 / K] * K
    sigma = [mat(np.identity(testdataMat.shape[1], dtype=np.float64)) for x in range(K)]
    gamma = mat(zeros((m, K)))
    startTime = time.time()
    pre_label = []
    Q = 0
    for i in range(maxIter):
        totals = []
        for j in range(m):
            sumAlphaMulP = 0
            for k in range(K):
                gamma[j, k] = alpha[k] * prob(testdataMat[j, :], mu[k], sigma[k])
                sumAlphaMulP += gamma[j, k]
            totals.append(sumAlphaMulP)
            for k in range(K):
                gamma[j, k] /= sumAlphaMulP
        sumGamma = sum(gamma, axis=0)
        for k in range(K):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = mat(zeros((n, n)))
            for j in range(m):
                mu[k] += gamma[j, k] * testdataMat[j, :]
            mu[k] /= sumGamma[0, k]
            for j in range(m):
                sigma[k] += gamma[j, k] * (testdataMat[j, :] - mu[k]).T * (testdataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]
            alpha[k] = sumGamma[0, k] / m
        ## step 1: init centroids
        clusterAssign = np.mat(np.zeros((m, 2)))
        for mm in range(m):
            # amx返回矩阵最大值，argmax返回矩阵最大值所在下标
            clusterAssign[mm, :] = np.argmax(gamma[mm, :]), np.amax(gamma[mm, :])  # 15.确定x的簇标记lambda
        stopTime = time.time()
        pre_label = [int(c) for c in clusterAssign[:, 0]]
        EMLabel.append(metrics.fowlkes_mallows_score(labelMat, pre_label))
        Q_new = np.sum(np.log(np.array(totals)))
        if abs(Q_new - Q) < 1:
            continue
        else:
            Q = Q_new
        evulate(testdataMat, labelMat, pre_label)
    return pre_label, EMLabel
