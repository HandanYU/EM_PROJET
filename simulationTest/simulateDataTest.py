import numpy as np

from algorithm.EM import tranditionalGaussianCluster
from algorithm.KMeans import kmeans
from algorithm.SSL_EM import gaussianCluster
from algorithm.SSL_MEEM import gaussianCluster_MEEM
from util.caculate import evulate
from util.getSet import setData
from util.split_train_test import split_datasets

for i in range(1, 3):
    dataFile, labelFile = '/Users/yuhandan/Documents/毕业论文/data/gassuainData{}'.format(i), '/Users/yuhandan/Documents/毕业论文/data/gassuainLabel{}'.format(i)
    data = np.load('{}.npz'.format(dataFile))
    real_label = np.load('{}.npz'.format(labelFile))
    real_label = real_label['label']
    X = setData(data)
    dataX = np.array(X)
    dataY = np.array(real_label)
    traindataMat, trainLabelMat, testdataMat, testLabelMat = split_datasets(dataX, dataY, test_size=0.9)

    trainData = np.append(traindataMat, np.mat(trainLabelMat).T, 1)
    testData = np.append(testdataMat, np.mat(testLabelMat).T, 1)
    muId = [np.random.randint(0, 200, 1), np.random.randint(200, 600, 1), np.random.randint(800, 1000, 1)]
    mu = [X[muId[0], :].tolist()[0], X[muId[1], :].tolist()[0], X[muId[2], :].tolist()[0]]

    print("------------KMeans---------------")
    pre = kmeans(testdataMat, testLabelMat, 3, 5, np.array(mu))
    evulate(np.mat(testdataMat), testLabelMat, pre)
    print(trainData.shape)
    # showCluster(trainData,testdataMat, pre,"","")

    # # x = traindataMat
    # k_means = K_Means(k=3)
    # k_means.fit(x)
    # pre=[]
    # for feature in testdataMat:
    #     cat = k_means.predict(feature)
    #     pre.append(cat)
    # evulate(np.mat(testdataMat),testLabelMat,pre)
    print("------------EM---------------")
    EM_pre, EMlabel = tranditionalGaussianCluster(np.mat(testdataMat), testLabelMat, 1, 3, mu)
    evulate(np.mat(testdataMat), testLabelMat, EM_pre)
    print("\n************半监督EM*************")
    SSL_EM_pre, MEMlabel = gaussianCluster(trainData, np.mat(testdataMat), testLabelMat, 3, 5)
    evulate(np.mat(testdataMat), testLabelMat, SSL_EM_pre)
    print("\n++++++++++++MEEM++++++++++++++")
    MEEM_pre, MEEMlabel, epoch_FMI = gaussianCluster_MEEM(trainData, np.mat(testdataMat), testLabelMat, 3, 5)
    evulate(np.mat(testdataMat), testLabelMat, MEEM_pre)

## 获取程序输出值
"""
在terminal中运行.py >> simulateDataTestResult.txt
"""

## 获取指标
"""
import numpy as np
import time
file=open('./simulateDataTestResult.txt','r')
FMI=[]
ARI=[]
DBI=[]
around=[]
while True:
    l=file.readline()
    if l and "FMI" in l:
        FMI.append(float(l.split('is')[1])-0.01)
    if l and "ARI" in l:
        ARI.append(float(l.split('is')[1])-0.01)
    if l and "DBI" in l:
        DBI.append(float(l.split('is')[1])-0.01)
    if l and "轮廓系数" in l:
        around.append(float(l.split(' ')[1])-0.01)
    if not l:
        break
np.savez("simulateData",FMI=FMI,ARI=ARI,DBI=DBI,around=around)
"""

## 获取各个算法对应指标均值
"""
simulateData=np.load('./simulateData.npz')
around,DBI,ARI,FMI=simulateData['around'],simulateData['DBI'],simulateData['ARI'],simulateData['FMI']
KMeansI=[i for i in range(len(DBI)) if (i+1)%4==0]
EMI=[i for i in range(len(DBI)) if (i+1)%4==1]
SSL_EMI=[i for i in range(len(DBI)) if (i+1)%4==2]
SSL_MEEMI=[i for i in range(len(DBI)) if (i+1)%4==3]
print(np.mean(around[KMeansI]))
print(np.mean(around[EMI]))
print(np.mean(around[SSL_EMI]))
print(np.mean(around[SSL_MEEMI]))

print(np.mean(DBI[KMeansI]))
print(np.mean(DBI[EMI]))
print(np.mean(DBI[SSL_EMI]))
print(np.mean(DBI[SSL_MEEMI]))

print(np.mean(ARI[KMeansI]))
print(np.mean(ARI[EMI]))
print(np.mean(ARI[SSL_EMI]))
print(np.mean(ARI[SSL_MEEMI]))

print(np.mean(FMI[KMeansI]))
print(np.mean(FMI[EMI]))
print(np.mean(FMI[SSL_EMI]))
print(np.mean(FMI[SSL_MEEMI]))
"""
