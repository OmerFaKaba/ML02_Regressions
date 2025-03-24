# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 05:17:30 2025

@author: omer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1)




from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
pol_reg = PolynomialFeatures(degree=4)

X_pol = pol_reg.fit_transform(X_train)

lr2 = LinearRegression()
lr2.fit(X_pol, y_train)


y_pred = lr2.predict(pol_reg.transform(X_test))
ComparePolynomialRegression = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)

#R Squared Score
from sklearn.metrics import r2_score
R2ScorePolynomialRegression = r2_score(y_test, y_pred)
