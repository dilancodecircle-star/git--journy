from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt

bc = load_breast_cancer()

x = scale(bc.data)
y = bc.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2)
model = KMeans(n_clusters = 2 , random_state = 0)
model.fit(x_train)
predict = model.predict(x_test)
accuracy = accuracy_score(y_test , predict)

print(f"accuracy : {accuracy}")
print(f"predict : {predict}")
print(f"acctual values : {y_test}")
print(f"labels:{model.labels_}")

plt.scatter(predict , y_test)
plt.show()