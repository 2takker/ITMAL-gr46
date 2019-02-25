import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix


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


def MNIST_PlotDigit(data, title = "", xlabel = "", ylabel = ""):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = 'Greys')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def IRIS_GetDataSet(return_X_y = False):
    return load_iris(return_X_y = return_X_y)


def IRIS_Plot(iris):
    colors = []
    for value in iris.target:
        if value == 0:
            colors.append('DarkBlue')
        elif value == 1:
            colors.append('DarkRed')
        elif value == 2:
            colors.append('Green')
            
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    scatter_matrix(df[iris.feature_names], figsize = (20, 20), s = 70, color = colors)
    
    plt.show()
