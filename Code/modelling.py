# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:37:37 2023

@author: sanket
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "feat_df.pkl"
feat_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)
feat_data2= feat_data[feat_data['date']>'2011/1/1']
feat_data2.drop('date', axis=1,inplace=True)
feat_data2.fillna(0,inplace=True)
#%%
label_encoder = LabelEncoder()
feat_data2['winner_le'] = label_encoder.fit_transform(feat_data2['winner_encoded'])
feat_data2.drop('winner_encoded',axis=1,inplace=True)

x_train,x_test, y_train, y_test= train_test_split(feat_data2.drop('winner_le',axis=1),feat_data2['winner_le'],test_size=0.2)
#%%
lr= LogisticRegression()
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
print(f1_score(y_test, y_pred))
#%%
rf= RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
print(f1_score(y_test, y_pred))
#%%
gbc= GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred= gbc.predict(x_test)
print(f1_score(y_test, y_pred))

