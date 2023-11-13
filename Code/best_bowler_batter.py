# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 01:20:28 2023

@author: sanket
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot 
#%%
icc_tournamet_winners_data= 'K:\\Sanket-datascience\\CWC_prediction\\Data\\icc_tournaments_winners_list.csv'
df_icc_winners= pd.read_csv(icc_tournamet_winners_data, encoding='unicode_escape')

all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\odis_male_csv2 (1)\\"
all_cwc_data_loc= 'K:\\Sanket-datascience\\CWC_prediction\\Data\\icc_mens_cricket_world_cup_male_csv2\\'

ls_odi_info_files= glob.glob(all_odi_data_loc+'*info.csv')
ls_cwc_info_files= glob.glob(all_cwc_data_loc+'*info.csv')
ls_odi_data_files= glob.glob(all_odi_data_loc+'*.csv')
ls_cwc_data_files= glob.glob(all_cwc_data_loc+'*.csv')

ls_odi_data_files= [i for i in ls_odi_data_files if i not in ls_odi_info_files]
ls_cwc_data_files= [i for i in ls_cwc_data_files if i not in ls_cwc_info_files]
ls_cwc_data_files= [i for i in ls_cwc_data_files if 'all_matches' not in i]
#%%
df_cwc_info= pd.DataFrame()
for i in ls_odi_data_files:
    print (i)
    df_info_single= pd.read_csv(i)
    df_info_single.reset_index(inplace=True, drop=True)
    df_cwc_info= pd.concat([df_info_single,df_cwc_info],ignore_index=True)
#%%
df_23wc_data= df_cwc_info[df_cwc_info['start_date']>='2023-10-05']
# df_23wc_data['wicket_type']=df_23wc_data['wicket_type'].fillna(0,inplace=False)
#%%
bowlers_wick_type= ['caught','lbw', 'bowled', 'caught and bowled','stumped',np.nan]
top_batters= df_23wc_data.groupby(['striker','batting_team'])['runs_off_bat'].sum().reset_index().sort_values('runs_off_bat',ascending=False).reset_index(drop=True).head(20)
top_bowlers= df_23wc_data[df_23wc_data['wicket_type'].isin(bowlers_wick_type)].groupby(['bowler','bowling_team'])['wicket_type'].count().reset_index().sort_values('wicket_type',ascending=False).reset_index(drop=True).head(20)
top_batters_nm= top_batters['striker'].values
top_bowlers_nm= top_bowlers['bowler'].values
#%%
runs_ts= df_23wc_data[df_23wc_data['striker'].isin(top_batters_nm)].groupby(['striker','match_id'])['runs_off_bat'].sum().reset_index().sort_values(['striker','match_id'])
wickets_ts= df_23wc_data[df_23wc_data['bowler'].isin(top_bowlers_nm)&(df_23wc_data['wicket_type'].isin(bowlers_wick_type))].groupby(['bowler','match_id'])['wicket_type'].count().reset_index().sort_values(['bowler','match_id'])
#%%
runs_ts['mat_no']=runs_ts.groupby('striker').cumcount()
wickets_ts['mat_no']=wickets_ts.groupby('bowler').cumcount()
#%%
runs_ts2= runs_ts.pivot(index='striker', columns= 'mat_no', values= 'runs_off_bat')
wickets_ts2= wickets_ts.pivot(index='bowler', columns= 'mat_no', values= 'wicket_type')
#%%
recent_wicks={'A Zampa':2,'JR Hazlewood':0,'Haris Rauf':3,'Shaheen Shah Afridi':2,'AU Rashid':2,'BFW de Leede':2,'LV van Beek':0,'PA van Meekeren':1,'JJ Bumrah':2,'Kuldeep Yadav':2,'Mohammed Shami':0,'RA Jadeja':2}
recent_runs= {'DA Warner':53,'DJ Malan':31,'Abdullah Shafique':0,'Mohammad Rizwan':36,'RG Sharma':61,'V Kohli':51}
#%%
for k,v in recent_wicks.items():wickets_ts2.loc[k,8]=v
for k,v in recent_runs.items():runs_ts2.loc[k,8]=v
#%%
runs_ts3 = runs_ts2.unstack().reset_index(name='value').sort_values(['striker','mat_no'])
wickets_ts3 = wickets_ts2.unstack().reset_index(name='value').sort_values(['bowler','mat_no'])
#%%
runs_ts3['runs_prev_mat'] = runs_ts3.groupby('striker')['value'].shift()
wickets_ts3['wicks_prev_mat'] = wickets_ts3.groupby('bowler')['value'].shift()
#%%
runs_ts3.dropna(subset=['runs_prev_mat','value'],inplace=True)
wickets_ts3.dropna(subset=['wicks_prev_mat','value'],inplace=True)
#%%
X = runs_ts3[['mat_no','value','runs_prev_mat']].values
train_size = int(len(X) * 0.8)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
 
# persistence model
def model_persistence(x):
 return x
 
# walk-forward validation
predictions = list()
for x in test_X:
 yhat = model_persistence(x)
 predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
