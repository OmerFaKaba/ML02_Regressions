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


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X_train, y_train)


#predict Attempt 
y_pred = regressor.predict(X_test)
CompareDecisionTreeRegression = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)



#R Squared Score
from sklearn.metrics import r2_score
R2ScoreDecisionTree = r2_score(y_test, y_pred)














