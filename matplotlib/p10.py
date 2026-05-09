import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
x = np.array([1,2,3,4,5])
figure , axis = plt.subplots(2,2)
axis[0,0].plot(x , x*2  )
axis[0,0].set_title("Red line" )

axis[0,1].bar(x , x**2  )
axis[0,1].set_title("Green line" )


axis[1,0].plot(x , x**3  ,)
axis[1,0].set_title("Red line" )

axis[1,1].plot(x , x**4 - x  )
axis[1,1].set_title("Green line" )
plt.show()