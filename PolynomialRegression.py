# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:25:26 2022

@author: siddh
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score
df = pd.read_csv('Boston.csv')


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

kfold = KFold(n_splits = 5,shuffle = True , random_state = 2022)
pf = PolynomialFeatures(degree = 2)

pl = pf.fit_transform(X)
pf.get_feature_names()

lr = LinearRegression()

cross_val_score(lr,pl,y,scoring='r2',cv=kfold).mean()
