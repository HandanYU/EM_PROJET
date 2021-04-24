import matplotlib.pyplot as plt


def show_origin_data(data):
    """
    ## 绘制原始数据分布
    :param data: 通过np.load('XX.npz')得到的包含多个类别数据的数据集
    :return:
    """
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    for cluster in data:
        x, y = data[cluster].T
        plt.scatter(x, y)


def showCluster(traindataMat, testdataMat, preLabel, saveName, titleName):
    """
    ## 绘制聚类结果
    :param traindataMat: 训练集，包括数据和标签
    :param testdataMat: 测试集，只有数据集
    :param preLabel: 预测标签
    :param saveName: 聚类图保存名称
    :param titleName: 图表题
    :return:
    """
    numSamples, dim = testdataMat.shape
    trainSamples = traindataMat.shape[0]
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    ## 绘制测试集样本点
    mark = ['+r', '+b', '+g', 'ok', 'or', '+r', 'sr', 'dr', '<r', 'pr']
    if len(set(preLabel)) > len(mark):
        print("Sorry! Your k is too large!")
        return 1
    for i in range(numSamples):
        markIndex = int(preLabel[i])
        plt.plot(testdataMat[i, 0], testdataMat[i, 1], mark[markIndex])

    ## 绘制训练集样本点
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(trainSamples):
        markIndex = int(traindataMat[i, -1])
        # print(markIndex)
        plt.plot(traindataMat[i, 0], traindataMat[i, 1], mark[markIndex])
    plt.title(titleName)
    # plt.savefig('./{}'.format(saveName))
    plt.show()
