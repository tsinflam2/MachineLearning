
# Rescale data (between 0 and 1)
import pandas
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Disable the ignorance of print list [0 0 0 0 ... 0 0 0 0] = > [0 0 0 0 0 100 0 100 -100 0 0]
np.set_printoptions(threshold=np.nan)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

print(X)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])