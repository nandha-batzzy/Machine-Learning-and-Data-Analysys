import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df =pd.read_csv('50_Startups.csv')

x =df.iloc[:,:-1]
y =df.iloc[:,4]

one_hot = pd.get_dummies(x['State'],drop_first= True)
X = x.drop('State',axis = 1)

X = pd.concat([X,one_hot],axis = 1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split()
