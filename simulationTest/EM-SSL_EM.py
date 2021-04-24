import numpy as np

from algorithm.EM import tranditionalGaussianCluster
from algorithm.SSL_EM import gaussianCluster
from conf.config import simulationData_path as path
from util.caculate import evulate
from util.getSet import setData
from util.split_train_test import split_datasets

FMI = {10: {'EM': [], 'MEM': []}, 15: {'EM': [], 'MEM': []}}


def getData(number):
    dataFile, labelFile = '{}/gassuainData{}'.format(path, 10), '{}/gassuainLabel{}'.format(path, number)  # 修改10，15
    data = np.load('{}.npz'.format(dataFile))
    real_label = np.load('{}.npz'.format(labelFile))
    real_label = real_label['label']
    X = setData(data)
    dataX = np.array(X)
    dataY = np.array(real_label)
    return X, dataX, dataY


for k in FMI.keys():
    X, dataX, dataY = getData(k)
    ## 进行随机40次sample的选择
    for i in range(40):
        traindataMat, trainLabelMat, testdataMat, testLabelMat = split_datasets(dataX, dataY)
        trainData = np.append(traindataMat, np.mat(trainLabelMat).T, 1)

        muId = [np.random.randint(0, 200, 1), np.random.randint(200, 600, 1), np.random.randint(600, 1000, 1)]
        mu = [X[muId[0], :].tolist()[0], X[muId[1], :].tolist()[0], X[muId[2], :].tolist()[0]]
        print("------------EM---------------")
        EM_pre, EMlabel = tranditionalGaussianCluster(np.mat(testdataMat), testLabelMat, 1, 3, mu)
        EM_FMI = evulate(np.mat(testdataMat), testLabelMat, EM_pre)[3]
        FMI[k]['EM'].append(EM_FMI)

        print("\n************半监督EM*************")
        SSL_EM_pre, MEMLabel = gaussianCluster(trainData, np.mat(testdataMat), testLabelMat, 3, 1)
        MEM_FMI = evulate(np.mat(testdataMat), testLabelMat, SSL_EM_pre)[3]
        FMI[k]['MEM'].append(MEM_FMI)
print(FMI)
