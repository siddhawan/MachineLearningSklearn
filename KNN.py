# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:56:28 2022

@author: siddh
"""


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import  accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('Sonar.csv')
df_new = pd.get_dummies(df,drop_first=True)
X = df_new.drop(['Class_R'],axis =1)
y = df_new.Class_R


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify=y,random_state = 2022)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
y_pred_prob = knn.predict_proba(X_test) 

area = roc_auc_score(y_test, y_pred_prob[:,-1])

print(f"At Neighbour 17 the area comes  = {area}")

