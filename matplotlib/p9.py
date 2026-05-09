import numpy as np
import matplotlib.pyplot as plt
score = np.random.normal(loc = 80 , scale=10 , size = 100)
score = np.clip(score , 0,100)
plt.hist(score , bins = 20 , 
                edgecolor = "Black")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tick_params(axis = "both")
plt.show()
