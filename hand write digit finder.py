import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist
from sklearn.datasets import fetch_openml


#train data

mnist_data = fetch_openml('mnist_784' , version = 1 , parser = 'auto')
x_train = mnist_data.data
y_train = mnist_data.target


print(x_train.shape , y_train.shape )