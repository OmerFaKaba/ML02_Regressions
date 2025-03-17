# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 06:44:11 2025

@author: omer
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
data = pd.read_csv("data.csv")

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


#missing values (no none values)


#Change variable with OneHotEncoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],
                       remainder="passthrough")

X = np.array(ct.fit_transform(X))



#Train and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


#Learning Multiple Linear Regression Model on Train Set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Testing model
y_predict = regressor.predict(X_test)

