import matplotlib.pyplot  as plt
import numpy as np

catg =np.array( ["Grains" , "Fruit" , "Vegitables" , "protien" , "Dairy" , "Sweets"])
value =np.array( [10,4,5,3,6,8])
plt.xlabel("type")
plt.ylabel("value")
plt.barh(catg , value , color = "red" )
plt.show()