# -*- coding: utf-8 -*-
"""
Created on Sun May  8 23:12:44 2022

@author: siddh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv("Sonar.csv")

le = LabelEncoder()
X = df.drop(['Class'],axis = 1)
y = df.Class
y = le.fit_transform(y)


knn = KNeighborsClassifier()
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

n_neigh = {'n_neighbors' : np.arange(1,101)}
ans = GridSearchCV(knn, n_neigh,scoring='neg_log_loss',cv = kfold)
ans.fit(X,y)
print(ans.best_params_)
print(ans.best_score_)
df1 = pd.DataFrame(ans.cv_results_)
