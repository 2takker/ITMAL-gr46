def dummy_func() -> 'string':
    return 'The dummy function has been called'

def dummy_func2():
    return 'This is a new dummy function'

def IRIS_Plot_2D(col1,col2):
    x_min, x_max = col1.min() - .5, col1.max() + .5
    y_min, y_max = col2.min() - .5, col2.max() + .5
    
    plt.scatter(col1, col2, c=y, cmap=plt.cm.Set1,edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    

import sklearn.datasets as sk
#from keras.datasets import mnist
import matplotlib.pyplot as plt



def IRIS_GetDataSet():
    iris= sk.load_iris()
    X = iris.data
    y = iris.target
    return X, y

from sklearn.datasets import fetch_openml
#from keras.datasets import mnist
import matplotlib.pyplot as plt

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.show()
    
    # TODO: add plot functionality for a single digit...

def MNIST_GetDataSet():
    mnist= fetch_openml('mnist_784', version=1, cache=True)
    # TODO: use mnist = fetch_mldata('MNIST original') or mnist.load_data(),
    #       but return as a single X-y tuple 
    X = mnist.data
    y = mnist.target
    return X, y

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

def MOON_GetDataSet():
    return make_moons()
   
def MOON_Plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y);
    plt.show()

