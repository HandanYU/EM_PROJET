import numpy as np
from numpy import *

from util.caculate import evulate


# 计算欧氏距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum((vector1 - vector2) ** 2))


# 初始化质心（初始化各个类别的中心点）
def initCentroids(data, k):
    numSample, dim = data.shape
    # k个质心
    centroids = np.zeros((k, dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引
        index = int(np.random.uniform(0, numSample))
        # 初始化质心
        centroids[i, :] = data[index, :]
    return centroids


# k-means算法函数
def kmeans(data, real_label, k, max_iter, mu):  # data为testdataMat,
    # 计算样本个数
    numSample = data.shape[0]
    # 保存样品属性（第一列保存该样品属于哪个簇，第二列保存该样品与它所属簇的误差（该样品到质心的距离））
    clusterData = np.array(np.zeros((numSample, 2)))
    # 确定质心是否需要改变
    # clusterChanged = True
    # 初始化质心
    centroids = mu
    id = 0
    while id < max_iter:
        id += 1
        # clusterChanged = False
        # 遍历样本
        for i in range(numSample):
            # 该样品所属簇（该样品距离哪个质心最近）
            minIndex = 0
            # 该样品与所属簇之间的距离
            minDis = 100000.0
            # 遍历质心
            for j in range(k):
                # 计算该质心与该样品的距离
                distance = euclDistance(np.array(centroids[j]), data[i, :])
                # 更新最小距离和所属簇
                if distance < minDis:
                    minDis = distance
                    clusterData[i, 1] = minDis
                    minIndex = j
            # 如果该样品所属的簇发生了改变，则更新为最新的簇属性，且判断继续更新簇
            if clusterData[i, 0] != minIndex:
                clusterData[i, 0] = minIndex
                # clusterChanged = True

        evulate(data, real_label, clusterData[:, 0])
        # 更新质心
        for j in range(k):
            # 获取样本中属于第j个簇的所有样品的索引
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            # 获取样本中于第j个簇的所有样品
            pointsInCluster = data[cluster_index]
            # 重新计算质心(取所有属于该簇样品的按列平均值)
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
    return clusterData[:, 0]


## 需要测试集的K_Means
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=10):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        self.centers_ = {}
        KMeansLabel = []
        for i in range(self.k_):
            self.centers_[i] = data[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            for i in range(self.k_):
                self.clf_[i] = []
            # print("质点:",self.centers_)
            for feature in data:
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum((cur_centers - org_centers) / org_centers * 100.0) > self.tolerance_:
                    optimized = False
            if optimized:
                break

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
