# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:14:27 2022

@author: siddh
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import  accuracy_score,confusion_matrix
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,cross_val_score


st = StandardScaler()

df = pd.read_csv('Image_Segmention.csv')
X = df.iloc[:,1:]
X_sc = st.fit_transform(X)
y= df.iloc[:,:1]
le = LabelEncoder()
y = le.fit_transform(y)

for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors = i)
    kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)
    result = cross_val_score(knn,X_sc,y,cv = kfold,scoring='neg_log_loss')
    print(result.mean())

# X_train,X_test,y_train,y_test = train_test_split(X_sc,y,test_size = 0.3,stratify=y,random_state = 2022)


# for i in range(1,50):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train,y_train)
#     y_pred = knn.predict(X_test)
#     y_pred_prob = knn.predict_proba(X_test) 
    
#     area = log_loss(y_test, y_pred_prob)
    
#     print(f"At Neighbour {i} the area comes  = {area}")
    

from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier(n_neighbors = i)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

n_neigh = {'n_neighbors' : np.arange(1,101)}
ans = GridSearchCV(knn, n_neigh,scoring='neg_log_loss',cv = kfold)
ans.fit(X_sc,y)
print(ans.best_params_)
print(ans.best_score_)
