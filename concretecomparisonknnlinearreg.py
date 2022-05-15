# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:30:57 2022

@author: siddh
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Concrete_Data.csv')
X = df.drop("Strength",axis =1)
sc = StandardScaler()
X_sc = sc.fit_transform(X)
y = df.Strength

knn = KNeighborsRegressor()
kfold = KFold(n_splits=5,random_state=2022,shuffle=True)
params = {'n_neighbors':np.arange(1,51)}
cv = GridSearchCV(estimator=knn, param_grid=params,cv=kfold,scoring='r2')
cv.fit(X_sc,y)
print(cv.best_params_)
print(cv.best_score_)
df1 = pd.DataFrame(cv.cv_results_)


lm = LinearRegression()
kfold = KFold(n_splits=5,shuffle=True,random_state=2022)
results = cross_val_score(lm, X,y,scoring='r2',cv=kfold)
print(results.mean())

