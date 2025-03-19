# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 05:13:07 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#Importing data
dataset = pd.read_csv("quality-price.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#Training od Decesion Tree Model
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)


regressor.fit(X, y)

#Atempt of Decision Tree Model Predict
y_predict = regressor.predict([[6.5]])


#Visualization
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))


plt.scatter(X, y, color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Decision Tree")
plt.xlabel("Qualitiy")
plt.ylabel("Price")
plt.show()