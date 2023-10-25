# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:37:37 2023

@author: sanket
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "feat_df.pkl"
feat_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)

label_encoder = LabelEncoder()
feat_data['winner_le'] = label_encoder.fit_transform(feat_data['winner_encoded'])
feat_data.drop('winner_encoded',axis=1,inplace=True)

x_train,x_test, y_train, y_test= train_test_split(feat_data.drop('winner_le',axis=1),feat_data['winner_le'],test_size=0.2)
#%%
lr= LogisticRegression()
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
f1_score(y_test, y_pred)
