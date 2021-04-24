import numpy as np
from numpy import *
from sklearn.cluster import KMeans

from algorithm.SSL_EM import gaussianCluster
from conf.config import simulationData_path as path
from util.getSet import setData
from util.show_scatter import showCluster

data = np.load('{}/gassuainData20.npz'.format(path))
label = np.load('{}/gassuainLabel20.npz'.format(path))['label']
X = setData(data)
trainData = np.append(X, np.mat(label).T, 1)
d1, d2, d3 = data['data1'], data['data2'], data['data3']

estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)
# print(estimator.cluster_centers_)

dis1 = np.array([np.linalg.norm(d1[i, :] - np.array([2.07398945, 5.04095436])) for i in range(d1.shape[0])])
dis2 = np.array([np.linalg.norm(d2[i, :] - np.array([2.97776672, 1.00980702])) for i in range(d2.shape[0])])
dis3 = np.array([np.linalg.norm(d3[i, :] - np.array([-1.09452158, 2.0786085])) for i in range(d3.shape[0])])

index1 = np.where(dis1 > 5.6)
index2 = np.where(dis2 > 2.1)
index3 = np.where(dis3 > 6.5)
#
index1 = index1[0].tolist()
index2 = index2[0].tolist()
index3 = index3[0].tolist()

traindataMat = np.append(np.append(d1[np.array(index1)], d2[np.array(index2)], 0), d3[np.array(index3)], 0)
print(len(index1), len(index2), len(index3))

## 异常点

# trainLabelMat=np.repeat(np.array([0,1,2]),[len(index1),len(index2),len(index3)])
# trainData=np.append(traindataMat,np.mat(trainLabelMat).T,1)
# randomIndex=index1+index2+index3
#
# testIndex=np.delete(np.array(range(X.shape[0])),randomIndex)
# testdataMat=X[testIndex,:]
# testLabelMat=label[testIndex]
#
# print("\n************半监督EM*************")
# pre_label,MEMLabel=gaussianCluster(trainData,np.mat(testdataMat),testLabelMat,3, 5)
# showCluster(trainData,np.mat(testdataMat), pre_label,"SSL_EM","SSL_EM")
#
# print("\n++++++++++++MEEM++++++++++++++")
# pre_label=gaussianCluster_MEEM(trainData,np.mat(testdataMat),testLabelMat,3,5)
# showCluster(trainData,np.mat(testdataMat), pre_label,"SSL_MEEM","SSL_MEEM")

# # 错误点
trainLabelMat = np.repeat(np.array([0, 1, 2]), [7, 9, 11])
print(traindataMat.shape)
print(np.mat(trainLabelMat).T.shape)
trainData = np.append(traindataMat, np.mat(trainLabelMat).T, 1)
randomIndex = index1 + index2 + index3
testIndex = np.delete(np.array(range(X.shape[0])), randomIndex)
testdataMat = X[testIndex, :]
testLabelMat = label[testIndex]
#
#
# print("\n************半监督EM*************")
pre_label, MEMLabel = gaussianCluster(trainData, np.mat(testdataMat), testLabelMat, 3, 5)
showCluster(trainData, np.mat(testdataMat), pre_label, "SSL_EM_1", "")

#
# #print("\n++++++++++++MEEM++++++++++++++")
# pre_label=gaussianCluster_MEEM(trainData,np.mat(testdataMat),testLabelMat,3,5)
# showCluster(trainData,np.mat(testdataMat), 3, pre_label,"SSL_MEEM_1","SSL_MEEM")
