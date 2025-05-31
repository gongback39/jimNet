import numpy as np
from sklearn.datasets import fetch_openml

def to_one_hot(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def MNIST_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype('int')

    X = X / 255.0
    
    y_onehot = to_one_hot(y, num_classes=10)
    
    return X, y_onehot