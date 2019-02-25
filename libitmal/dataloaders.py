import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris


def MOON_GetDataSet(n_samples = 100, shuffle = True, noise = None,
                   random_state = None):
    return make_moons(n_samples = n_samples, shuffle = shuffle, 
                      noise = noise, random_state = random_state)

   
def MOON_Plot(X, y, title = "", xlabel = "", ylabel = ""):
    plt.scatter(X[:,0], X[:,1], c = y)   
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()


def MNIST_GetDataSet():
    mnist = fetch_openml('mnist_784', version = 1, cache = True)
    return mnist.data, mnist.target


def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = 'Greys')


def IRIS_GetDataSet(return_X_y = False):
    return load_iris(return_X_y = return_X_y)
