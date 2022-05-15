# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:49:18 2022

@author: siddh
"""


import pandas as pd 
from sklearn.model_selection import train_test_split ,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import numpy as np

df = pd.read_csv('bank.csv' ,sep = ';')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X = pd.get_dummies(X,drop_first=True)
y = pd.get_dummies(y,drop_first=True)

kfold = StratifiedKFold(n_splits = 5,shuffle = True, random_state = 2022)

cross_val_score(LogisticRegression(),X,y,scoring = 'roc_auc',cv = kfold).mean()
cross_val_score(GaussianNB(),X,y,scoring = 'roc_auc',cv = kfold).mean()
