# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:19:49 2022

@author: siddh
"""

import pandas as pd 
from sklearn.model_selection import train_test_split ,KFold,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
df = pd.read_csv("Boston.csv")
X = df.drop('medv',axis = 1)
y = df['medv']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2022) 

ridge = Ridge()
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)

print(r2_score(y_test,y_pred))

params = {'alpha' : np.linspace(0,10,20)}
kfold = KFold(n_splits = 5 , shuffle = True,random_state = 2022)
results  = GridSearchCV(ridge,params,scoring = 'r2', cv = kfold)
results.fit(X,y)

print(results.best_params_)
print(results.best_score_)
