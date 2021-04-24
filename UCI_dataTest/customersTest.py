### 对数据集Wholesale_customers的处理，并测试
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from UCI_dataTest.UCI_Test import test
from conf.config import UCIdata_path as path

data = pd.read_csv('{}/Wholesale_customers.csv'.format(path), header=0, sep=',')
dataY = data['Channel']

# 删除'Region', 'Channel'特征
data.drop(['Region', 'Channel'], axis=1, inplace=True)
print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
pd.scatter_matrix(data, alpha=0.3, figsize=(14, 8), diagonal='kde')

### 拷贝数据集
indices = [3, 141, 340]  # 样本的索引
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)  # 创建samples保存样本数据

data_copy = data.copy()
samples_copy = samples.copy()

### 应用boxcox变换
for feature in data_copy:
    data_copy[feature] = stats.boxcox(data_copy[feature])[0]
log_data = data_copy

for feature in data:
    samples_copy[feature] = stats.boxcox(samples_copy[feature])[0]
log_samples = samples_copy

# 画图
pd.scatter_matrix(data_copy, alpha=0.3, figsize=(14, 8), diagonal='kde');

### 异常值处理
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)

    print("特征'{}'的异常值包括:".format(feature))

outliers = [95, 338, 86, 75, 161, 183, 154]  # 选择需要删除的异常值

dataX = log_data.drop(log_data.index[outliers]).reset_index(drop=True)  # 删除选择的异常值
dataY = dataY.drop(dataY.index[outliers]).reset_index(drop=True)

### 归一化处理
scaler = MinMaxScaler()
scaler.fit(dataX)
X = scaler.transform(dataX)
dataX = X
dataY = np.array(dataY)

label1, label2, label3, label4 = test(dataX, dataY)
