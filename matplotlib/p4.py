import matplotlib.pyplot  as plt
import numpy as np
x = np.array([11,21,31,41,51])
y= np.array([12,4,56,34,67])

x1 = np.array([12,22,32,42,52])
y1= np.array([1,4,5,3,6])
plt.xlabel("student" , fontsize = 10)
plt.tick_params(axis ="both")
plt.xticks(x)


plt.plot(x,y)
plt.plot(x1,y1)
plt.show()