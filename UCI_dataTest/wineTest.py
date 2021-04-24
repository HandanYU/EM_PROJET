import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from UCI_Test import test

wines = datasets.load_wine()
wineData = np.mat(wines.data)
wineTarget = np.mat(wines.target).T
dataY = wineTarget.tolist()
dataY = np.array([i[0] for i in dataY])

scaler = MinMaxScaler()
scaler.fit(wineData)
wineData = scaler.transform(wineData)
pca = PCA(n_components=5)
dataX = pca.fit_transform(wineData)

test(dataX, dataY)
