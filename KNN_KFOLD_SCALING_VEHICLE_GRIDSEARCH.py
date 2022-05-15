# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:34:25 2022

@author: siddh
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("Vehicle.csv")
X = df.drop(['Class'],axis = 1)
y = df.Class


sc = StandardScaler()
le = LabelEncoder()
X = sc.fit_transform(X)
y = le.fit_transform(y)
print(le.classes_)

knn = KNeighborsClassifier()
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

n_neigh = {'n_neighbors' : np.arange(1,101)}
ans = GridSearchCV(knn, n_neigh,scoring='neg_log_loss',cv = kfold)
ans.fit(X,y)
print(ans.best_params_)
print(ans.best_score_)
df1 = pd.DataFrame(ans.cv_results_)
