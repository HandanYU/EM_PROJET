import numpy as np


def create_two_uniform_datasets(low_list, high_list, size_list, k):
    """

    :param low_list: 每个均匀分布最小值列表
    :param high_list: 每个均匀分布最大值列表
    :param size_list: 每个均匀分布数据集大小列表 [( 200,2),( 100,2),( 300,2)]
    :param k: 均匀分布个数
    :return:
    """
    data = np.array([])
    for i in range(k):
        cluster = np.random.uniform(low_list[i], high_list[i], size_list[i])
        data = np.append(data, cluster, 0)
    return data


## 产生二维均匀分布
# 1、使用均匀分布函数随机三个簇，每个簇周围10个数据样本。
# cluster1 = np.random.uniform(1.5, 2.5, ( 300,2))
# cluster2 = np.random.uniform(5.5, 7.5, ( 300,2))
# cluster3 = np.random.uniform(2.0, 6.0, ( 300,2))

def create_two_normal_datasets(mu, sigma, size, k):
    """

    :param mu: 均值列表
    :param sigma: 方差列表
    :param size: 数据大小
    :param k: 数据集个数
    :return:
    """
    data = np.array([])
    for i in range(k):
        cluster = np.array([np.random.normal(mu[i], sigma[i], size=[1, 2]).tolist()[0] for _ in range(size[i])])
        data = np.append(data, cluster, 0)
        return data
# a=np.array([np.random.normal(1,0.3,size=[1, 2]).tolist()[0] for _ in range(300)])
# a1=np.array([np.random.normal(3,0.7,size=[1, 2]).tolist()[0] for _ in range(300)])
# a2=np.array([np.random.normal(-1.5,1,size=[1, 2]).tolist()[0] for _ in range(300)])

# for i in range(1, 41):
#     create_multi_normal_datasets('./data/gassuainData{}'.format(i), './data/gassuainLabel{}'.format(i))

# import matplotlib.pyplot as plt
# def show_scatter(data):
#     plt.axis()
#     plt.title("scatter")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     for d in data:
#         x, y = data[d].T
#         plt.scatter(x, y)
#
#
# # data = np.load('./data/gassuainData20.npz')
# # real_label = np.load('./data/gassuainLabel20.npz')
# # show_scatter(data)
