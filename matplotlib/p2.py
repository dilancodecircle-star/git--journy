import matplotlib.pyplot as plt
import numpy as np
x =np.array( [1,2,3,4,5])
y = np.array([10,20,15,40,2])
y2 = np.array([20 , 10 ,4, 16 , 8])
plt.title("Atheena code")
plt.plot(x,y , marker = ".",
         markersize = 10, 
         markerfacecolor = "black" , 
         linestyle = " ", 
         linewidth = 2,
          
               )

plt.plot(x, y2)
plt.show()
plt.plot(x, y2, marker = "o", markersize = 10, markerfacecolor = "red", linestyle = " ", linewidth = 2)
plt.show()