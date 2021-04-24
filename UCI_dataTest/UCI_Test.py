import numpy as np
from sklearn import metrics

from algorithm.EM import KMeans_EM
from algorithm.KMeans import K_Means
from algorithm.SSL_EM import gaussianCluster
from algorithm.SSL_MEEM import gaussianCluster_MEEM
from util.split_train_test import split_datasets


def test(dataX, dataY):
    label1 = []
    label2 = []
    label3 = []
    label4 = []

    for iter in range(10):
        kmeansLabel = []
        print("round: ", iter + 1)

        traindataMat, trainLabelMat, testdataMat, testLabelMat = split_datasets(dataX, dataY, 0.9)
        trainData = np.append(traindataMat, np.mat(trainLabelMat).T, 1)

        for id in range(20):
            k_means = K_Means(k=2, tolerance=0.0001, max_iter=id)
            k_means.fit(traindataMat)
            pre_kmeans = []
            for feature in testdataMat:  # 测试集
                pre_kmeans.append(k_means.predict(feature))
            kmeansLabel.append(metrics.fowlkes_mallows_score(testLabelMat, pre_kmeans))
        label1.append(kmeansLabel)

        print("+++++EM+++++++++")
        pre, EMLabel = KMeans_EM(traindataMat, np.mat(testdataMat), testLabelMat, 20, 2)
        label2.append(EMLabel)

        print("+++++++++MEM++++++++++")
        pre, MEMLabel = gaussianCluster(trainData, np.mat(testdataMat), testLabelMat, 2, 20)
        label3.append(MEMLabel)

        print("+++++++++++MEEM++++++++++")
        pre, MEEMLabel, epoch_FMI = gaussianCluster_MEEM(trainData, np.mat(testdataMat), testLabelMat, 2, 20)
        label4.append(MEEMLabel)
    return label1, label2, label3, label4
