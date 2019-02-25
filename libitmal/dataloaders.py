import sklearn 
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

def MOON_GetDataSet(n_samples = 100,noise = 0):
    # TODO: your code here...
    return sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)

def MOON_Plot(X, y,title="my title", xlable="", ylabel=""):
    # TODO: some simple plot commands, 
    # To plot pretty figures
    colors = ['#FF0000' if x == 1 else '#0000ff' for x in y]
    plt.title(title)
    plt.xlabel = xlable
    plt.ylabel = ylabel
    plt.scatter(X[:,0],X[:,1],s = 1,color=colors)
    plt.show()

from sklearn.datasets import fetch_mldata
from tensorflow.keras.datasets import mnist

def MNIST_PlotDigit(data):
    image = data.reshape(28, 28)
    # TODO: add plot functionality for a single digit...
    plt.imshow(image,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")

def MNIST_GetDataSet():
    # TODO: use mnist = fetch_mldata('MNIST original') or mnist.load_data(),
    #       but return as a single X-y tuple     
   
    (trainx,trainy), (testx,testy) = mnist.load_data()
    
    xset = np.concatenate((trainx, testx))
    yset = np.concatenate((trainy,testy))
    
    return xset, yset

def IRIS_GetDataSet():
	return sklearn.datasets.load_iris()

def IrisFeatureComparePlot(data,feature_nOne=0,feature_nTwo=0):
	#prepare colors
    for x in data.target:
        if x == 0:
             colors.append('DarkBlue')
        elif x == 1:
             colors.append('DarkRed')
        elif x == 2:
             colors.append('Green')

    plt.scatter(data.data[:,feature_nOne],data.data[:,feature_nTwo],s=1,color=colors)