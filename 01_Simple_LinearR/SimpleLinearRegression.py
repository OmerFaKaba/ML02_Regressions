# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 04:39:45 2025

@author: omer
"""
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("deneyim-maas.csv", sep=";")

plt.scatter(data.deneyim,data.maas)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Exp-Salary Graph")
plt.show()
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing data set
dataset = pd.read_csv("deneyim-maas.csv", sep=";")


X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values
 
 
#train and test set
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
 
 
#training slr on sets
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train) 

#testing on sets
y_predict = lr.predict(X_test)
 

#visulization result of training sets
plt.scatter(X_train, y_train, color="red")

plt.plot(X_train,lr.predict(X_train),color="blue")

plt.title("Exp-Salary Graph")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
 
#visualization reslt of test sets
plt.scatter(X_test, y_test, color="red")

plt.plot(X_train,lr.predict(X_train),color="blue")

plt.title("Exp-Salary Graph")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show() 
 
 
 
 
 
 
 
 
 
 
 