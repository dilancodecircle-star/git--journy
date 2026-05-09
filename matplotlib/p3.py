import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x1 = np.array([12,14,15,20,30])
y1 = np.array([2,34,56,22,14])
x = pd.Series([1,2,3,4,5,6])
y =pd.Series([12,45,67,45,23,45])
plt.plot(x,y)
plt.plot(x1,y1)
plt.show()