# -*- coding: utf-8 -*-
"""
Created on Sat May 14 20:11:47 2022

@author: siddh
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

df = pd.read_csv("BreastCancer.csv")
dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,1:-1]
y = dum_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2022,
                                                 test_size=0.3,
                                                 stratify=y)
scaler = MinMaxScaler()
model = SGDClassifier(loss='log',random_state=2022)
pipe = Pipeline([('scaler', scaler), ('SGD', model)])
pipe.fit(X_train, y_train)

y_pred_prob = pipe.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))
########### Grid Search CV #####################
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
params = {'SGD__eta0':np.linspace(0.0001,0.8,7),
          'SGD__learning_rate':['constant','optimal',
                           'invscaling','adaptive']}
gcv = GridSearchCV(pipe,scoring='roc_auc',cv=kfold,param_grid=params)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
