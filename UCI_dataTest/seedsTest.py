import numpy as np

from UCI_Test import test
from conf.config import UCIdata_path as path
from util.getSet import readTXT

data = np.mat(readTXT('{}/seeds_dataset.txt'.format(path)))
dataX, dataY = np.array(data[:, :-1]), np.array(data[:, -1])
dataY = np.array([int(i[0]) for i in dataY])
test(dataX, dataY)
