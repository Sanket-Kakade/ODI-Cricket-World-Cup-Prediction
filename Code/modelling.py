# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:37:37 2023

@author: sanket
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score
#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "feat_df.pkl"
feat_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)
feat_data2= feat_data[feat_data['date']>'2015/1/1']
feat_data2.drop('date',axis=1,inplace=True)

feat_data2.fillna(0,inplace=True)
#%%
label_encoder = LabelEncoder()
feat_data2['winner_le'] = label_encoder.fit_transform(feat_data2['winner_encoded'])
feat_data2.drop('winner_encoded',axis=1,inplace=True)
dummy_cols= ['team_1_Afghanistan', 'team_1_Australia',
'team_1_Bangladesh', 'team_1_Bermuda', 'team_1_Canada',
'team_1_England', 'team_1_Hong Kong', 'team_1_India', 'team_1_Ireland',
'team_1_Jersey', 'team_1_Kenya', 'team_1_Namibia', 'team_1_Nepal',
'team_1_Netherlands', 'team_1_New Zealand', 'team_1_Oman',
'team_1_Pakistan', 'team_1_Papua New Guinea', 'team_1_Scotland',
'team_1_South Africa', 'team_1_Sri Lanka',
'team_1_United Arab Emirates', 'team_1_United States of America',
'team_1_West Indies', 'team_2_Australia',
'team_2_Bangladesh', 'team_2_Bermuda', 'team_2_Canada',
'team_2_England', 'team_2_Hong Kong',
'team_2_India', 'team_2_Ireland', 'team_2_Kenya', 'team_2_Namibia',
'team_2_Nepal', 'team_2_Netherlands', 'team_2_New Zealand',
'team_2_Oman', 'team_2_Pakistan', 'team_2_Papua New Guinea',
'team_2_Scotland', 'team_2_South Africa', 'team_2_Sri Lanka',
'team_2_United Arab Emirates', 'team_2_United States of America',
'team_2_West Indies', 'team_2_Zimbabwe']
feat_data2.drop(dummy_cols, axis=1,inplace=True)
#%%
x_train,x_test, y_train, y_test= train_test_split(feat_data2.drop('winner_le',axis=1),feat_data2['winner_le'],test_size=0.2)
# train, test= feat_data2[feat_data2['date']<='2020/1/1'], feat_data2[feat_data2['date']>'2020/1/1']
# train.drop('date',axis=1,inplace=True)
# test.drop('date',axis=1,inplace=True)
# x_train,y_train=  train.drop('winner_le',axis=1), train[['winner_le']]
# x_test, y_test= test.drop('winner_le',axis=1), test[['winner_le']]

#%%
lr= LogisticRegression(max_iter=10000)
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
y_train_pred= lr.predict(x_train)
print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))

print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#%%
rf= RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
y_train_pred= rf.predict(x_train)

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))

print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#%%
gbc= GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred= gbc.predict(x_test)
y_train_pred= gbc.predict(x_train)

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#%%
svc= SVC()
svc.fit(x_train, y_train)
y_pred= svc.predict(x_test)
y_train_pred= svc.predict(x_train)

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#%%
# feature_names = [x_train.columns]
# importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
# forest_importances = pd.Series(importances, index=feature_names)
# forest_importances.sort_values(inplace=True,ascending=False)
# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
