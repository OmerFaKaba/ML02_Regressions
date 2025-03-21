# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:40:21 2025

@author: omer
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



dataset = pd.read_csv("quality-price.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Random Forest Model Training
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(random_state=0,n_estimators = 10)

regressor.fit(X, y)


#Random forest predict atempt
y_predict_random_forest = regressor.predict([[6.5]])


#Visiualization
X_grid = np.arange(min(X),max(X),0.1)
x_grid = X_grid.reshape((len(X_grid), 1))


plt.scatter(X, y,color="red")
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("Random Forest Model")
plt.xlabel("quality")
plt.ylabel("price")
plt.show()