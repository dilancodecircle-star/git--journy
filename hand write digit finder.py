import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist

#train data
x_train = mnist.train_image()
y_train = mnist.train_label()