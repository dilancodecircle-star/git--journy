import matplotlib.pyplot as plt
import numpy as np
x =[1,2,3,4,5]
y =[5,10,15,20,25]
plt.plot(x , y , marker = "." , markersize = 5
         )
plt.xticks(x)

plt.grid(axis = "both" , 
         linestyle = "--")
plt.show()