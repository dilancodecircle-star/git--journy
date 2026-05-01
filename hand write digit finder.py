import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist
from sklearn.datasets import fetch_openml




mnist_data = fetch_openml('mnist_784' , version = 1 , parser = 'auto')
x = mnist_data.data
y = mnist_data.target

#train data
x_train = x[:60000] #grabe the first 60000 data for training
y_train = y[:60000]

#test data

x_test = x[60000:]
y_test = y[60000:]
