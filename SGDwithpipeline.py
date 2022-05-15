# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:26:03 2022

@author: siddh
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


df = pd.read_csv("bank.csv",sep = ';')
dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

scaler = MinMaxScaler()
model = SGDClassifier(loss='log_loss',random_state=2022)
pipe = Pipeline([('scaler', scaler), ('SGD', model)])

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'SGD__eta0':np.linspace(0.0001,0.8,10),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive']}
gcv = GridSearchCV(pipe,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
gdcsv = pd.DataFrame(gcv.cv_results_)
