# 获取数据集
import numpy as np


def setData(data):
    X = np.array([[0, 0]])
    for d in data:
        X = np.append(X, data[d], 0)
    return np.mat(np.delete(X, 0, axis=0))


## 读取txt文件
def readTXT(path, delimiter='\t'):  # delimiter是数据分隔符
    fp = open(path, 'r')
    string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
    fp.close()
    row_list = string.splitlines()  # splitlines默认参数是‘\n’
    data_list = [[float(i) for i in row.strip().split(delimiter)] for row in row_list]
    return np.array(data_list)
