# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:56:53 2022

@author: siddh
"""

import pandas as pd
df = pd.read_csv('Concrete_Data.csv')
X = df.drop(['Strength'],axis = 1)
X.corr()
import seaborn as sns
sns.heatmap(X.corr(),annot =True)
