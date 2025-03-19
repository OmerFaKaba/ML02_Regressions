# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 03:33:11 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("quality-price.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Linear Regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X, y) 

y_linear_predict = lr.predict(X)

#Polynomial (linear) Regression Model Learning
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)

X_pol = pol_reg.fit_transform(X)


lr_pol = LinearRegression()
lr_pol.fit(X_pol, y)




#Visiualization of Linear Model
"""
plt.scatter(X,y,color="red")
plt.plot(X,lr.predict(X),color="blue")
plt.title("quality-price graph")
plt.xlabel("quality")
plt.ylabel("price")
plt.show()"""


#Visualization of Polynomial Model
plt.scatter(X,y,color="red")
plt.plot(X,lr_pol.predict(X_pol),color="blue")
plt.title("quality-price graph (Polynomial) Degree = 4")
plt.xlabel("quality")
plt.ylabel("price")
plt.show()


#Attempt of Linear Model Predict
linear_perdict = lr.predict([[8.5]])


#Atempt of Polynomial Model Predict
polynomial_predict = lr_pol.predict(pol_reg.fit_transform([[8.5]]))
