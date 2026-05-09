import matplotlib.pyplot as plt
import numpy as np
x1 = np.array([0,1,2,3,4,5,5,6])
y1 = np.array([20,22,24,26,28,30,32,34])

x2 = np.array([0,1,3,4,4,5,7,8])
y2 = np.array([20,22,26,28,38,40,80,88])

plt.scatter(x1 , y1 , color  ="skyblue" , label = "class A")
plt.scatter(x2 , y2 , color  ="orange" , label = "class B")
plt.legend()
plt.show()