import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')
print(df.describe())
print(df.info())
import sklearn 
import matplotlib.pyplot as plt
#sns.countplot(x='Survived', data=df)
a = pd.dframe