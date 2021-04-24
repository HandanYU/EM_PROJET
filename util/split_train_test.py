from sklearn.model_selection import StratifiedShuffleSplit


def split_datasets(dataX, dataY, test_size=0.9):
    """
    # 根据dataY取值类别数均匀随机划分训练集和测试集
    :param dataX: 数据X，为array类型
    :param dataY: 标签集，为array类型
    :return: 训练集和测试集
   """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, test_index in split.split(dataX, dataY):
        traindataMat, testdataMat = dataX[train_index], dataX[test_index]
        trainLabelMat, testLabelMat = dataY[train_index], dataY[test_index]

    return traindataMat, trainLabelMat, testdataMat, testLabelMat
