import numpy as np
import matplotlib.pyplot as plt
catg = ["Freshers" , "Sophmores" , "Juniors" , "Seniors"]
value = [300 , 250 , 275 ,225]
color = ["Red" , "Green"  , "Blue" , "yellow" ]
plt.title("University students")
plt.pie(value  , labels = catg 
            ,    autopct = "%1.1f%%"
            ,     colors = color , 
                    explode  =[0,0,0,0.1] , 
                    shadow = True , 
                    startangle = 180)
plt.show()