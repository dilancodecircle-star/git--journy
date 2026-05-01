import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist
from sklearn.datasets import fetch_openml




mnist_data = fetch_openml('mnist_784' , version = 1 , parser = 'auto')
x = np.array(mnist_data.data)
y = np.array(mnist_data.target)

#train data
x_train = x[:60000] #grabe the first 60000 data for training
y_train = y[:60000]

#test data

x_test = x[60000:]
y_test = y[60000:]

print(f"x_train: {x_train.shape}")
print(f"x_test: {x_test.shape}")
x_train = x_train/255
x_test = x_test/255

clf = MLPClassifier(solver = 'adam' , activation = 'relu' , hidden_layer_sizes=(64, 64))
clf.fit(x_train , y_train)
