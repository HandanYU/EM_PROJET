import numpy as np
from numpy import *

from algorithm.SSL_MEEM import gaussianCluster_MEEM
from conf.config import simulationData_path as path
from util.getSet import setData
from util.split_train_test import split_datasets

FMI = {10: [], 20: [], 30: []}


def cal_epoch_FMI(dataX, dataY, dataGroup):
    print("data: ", dataGroup)
    for i in linspace(1, 3, 20):
        print("delta: ", i)
        traindataMat, trainLabelMat, testdataMat, testLabelMat = split_datasets(dataX, dataY)
        trainData = np.append(traindataMat, np.mat(trainLabelMat).T, 1)
        pre_label, MEEM_label, echo_FMI = gaussianCluster_MEEM(trainData, np.mat(testdataMat), testLabelMat, 3, 10, i)
        FMI[dataGroup].append(echo_FMI)
    return


for k in FMI.keys():
    dataFile, labelFile = '{}/gassuainData{}'.format(path, k), '{}/gassuainLabel{}'.format(path, k)  ##需要修改
    data = np.load('{}.npz'.format(dataFile))
    real_label = np.load('{}.npz'.format(labelFile))
    real_label = real_label['label']
    X = setData(data)
    dataX = np.array(X)
    dataY = np.array(real_label)
    cal_epoch_FMI(dataX, dataY, k)
print(FMI)
