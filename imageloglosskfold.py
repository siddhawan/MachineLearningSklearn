# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:21:46 2022

@author: siddh
"""

import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('Image_Segmention.csv')

X = df.drop(['Class'],axis = 1)
#sc = StandardScaler()
#X = sc.fit_transform(X)
le = LabelEncoder()
y = df.Class
y = le.fit_transform(y)
gaussianNB = GaussianNB()

skfold = StratifiedKFold(n_splits = 5,shuffle=True,random_state = 2022)

score = cross_val_score(gaussianNB ,X,y ,scoring='neg_log_loss',cv=skfold)

print(score.mean())
