from PIL import Image
img = Image.open("7.png") 
data  = list(img.getdata())



for i in range(len(data)):
    data[i] = max(data) - data[i]

print(data)
 
