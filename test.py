from sklearn import preprocessing
import numpy as np
from skimage import transform,data
if __name__ == '__main__':
    X = np.array([[ 0., 1.,  2.],
    [ -1.,  1.,  3.],
    [ -3.,  1., 5.]])
    # mean = np.mean(X,axis=1)
    # mean = np.reshape(mean,[-1,1])
    # X_scaled = (X-mean)/mean
    # xx = np.delete(X_scaled,[1,2],axis=1)
    # print(X_scaled.shape)
    xx = np.reshape(X,[-1,9])

