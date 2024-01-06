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

all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\odis_male_csv\\"
# all_cwc_data_loc= 'K:\\Sanket-datascience\\CWC_prediction\\Data\\icc_mens_cricket_world_cup_male_csv2\\'

ls_odi_info_files= glob.glob(all_odi_data_loc+'*info.csv')
# ls_cwc_info_files= glob.glob(all_cwc_data_loc+'*info.csv')
ls_odi_data_files= glob.glob(all_odi_data_loc+'*.csv')
# ls_cwc_data_files= glob.glob(all_cwc_data_loc+'*.csv')

ls_odi_data_files= [i for i in ls_odi_data_files if i not in ls_odi_info_files]
# ls_cwc_data_files= [i for i in ls_cwc_data_files if i not in ls_cwc_info_files]
# ls_cwc_data_files= [i for i in ls_cwc_data_files if 'all_matches' not in i]
#%%
df_cwc_info= pd.DataFrame()
for i in ls_odi_data_files:
    # print (i)
    df_info_single= pd.read_csv(i)
    df_info_single.reset_index(inplace=True, drop=True)
    df_cwc_info= pd.concat([df_info_single,df_cwc_info],ignore_index=True)
#%%
df_23wc_data= df_cwc_info[df_cwc_info['start_date']>='2023-10-05']
# df_23wc_data['wicket_type']=df_23wc_data['wicket_type'].fillna(0,inplace=False)
#%%
bowlers_wick_type= ['caught','lbw', 'bowled', 'caught and bowled','stumped',np.nan]
top_batters= df_23wc_data.groupby(['striker'])['runs_off_bat'].sum().reset_index().sort_values('runs_off_bat',ascending=False).reset_index(drop=True).head(20)
top_bowlers= df_23wc_data[df_23wc_data['wicket_type'].isin(bowlers_wick_type)].groupby(['bowler'])['wicket_type'].count().reset_index().sort_values('wicket_type',ascending=False).reset_index(drop=True).head(20)
top_batters_nm= top_batters['striker'].values
top_bowlers_nm= top_bowlers['bowler'].values
#%%
runs_ts= df_23wc_data[df_23wc_data['striker'].isin(top_batters_nm)].groupby(['striker','match_id'])['runs_off_bat'].sum().reset_index().sort_values(['striker','match_id'])
wickets_ts= df_23wc_data[df_23wc_data['bowler'].isin(top_bowlers_nm)&(df_23wc_data['wicket_type'].isin(bowlers_wick_type))].groupby(['bowler','match_id'])['wicket_type'].count().reset_index().sort_values(['bowler','match_id'])
#%%
runs_ts['mat_no']=runs_ts.groupby('striker').cumcount()
wickets_ts['mat_no']=wickets_ts.groupby('bowler').cumcount()
#%%
runs_ts2= runs_ts.pivot(index=['striker'], columns= 'mat_no', values= 'runs_off_bat')
wickets_ts2= wickets_ts.pivot(index=['bowler'], columns= 'mat_no', values= 'wicket_type')
#%%
recent_wicks={'A Zampa':2,'JR Hazlewood':0,'Haris Rauf':3,'Shaheen Shah Afridi':2,'AU Rashid':2,'BFW de Leede':2,'LV van Beek':0,'PA van Meekeren':1,'JJ Bumrah':2,'Kuldeep Yadav':2,'Mohammed Shami':0,'RA Jadeja':2}
recent_runs= {'DA Warner':53,'DJ Malan':31,'Abdullah Shafique':0,'Mohammad Rizwan':36,'RG Sharma':61,'V Kohli':51}

recent_wicks_sf={'A Zampa':0,'AU Rashid':np.nan,'BFW de Leede':np.nan,'D Madushanka':np.nan,'G Coetzee':np.nan,'Haris Rauf':np.nan,'JJ Bumrah':1,'JR Hazlewood':2,'K Rabada':1,'KA Maharaj':1,'Kuldeep Yadav':1,'LV van Beek':np.nan,'M Jansen':0,'MJ Henry':np.nan,'MJ Santner':np.nan,'Mohammed Shami':7,'PA van Meekeren':np.nan,'RA Jadeja':0,'Shaheen Shah Afridi':np.nan,'TA Boult':1}
recent_runs_sf={'AK Markram':10,'Abdullah Shafique':np.nan,'Azmatullah Omarzai':np.nan,'DA Warner':29,'DJ Malan':np.nan,'DJ Mitchell':134,'DP Conway':13,'GJ Maxwell':1,'H Klaasen':47,'HE van der Dussen':6,'Hashmatullah Shahidi':np.nan,'Ibrahim Zadran':np.nan,'Mohammad Rizwan':np.nan,'P Nissanka':np.nan,'Q de Kock':3,'R Ravindra':13,'RG Sharma':47,'Rahmat Shah':np.nan,'S Samarawickrama':np.nan,'V Kohli':117}
#%%
for k,v in recent_wicks.items():wickets_ts2.loc[k,8]=v
for k,v in recent_runs.items():runs_ts2.loc[k,8]=v

for k,v in recent_wicks_sf.items():wickets_ts2.loc[k,9]=v
for k,v in recent_runs_sf.items():runs_ts2.loc[k,9]=v
#%%
runs_ts3 = runs_ts2.unstack().reset_index(name='value').sort_values(['striker','mat_no'])
wickets_ts3 = wickets_ts2.unstack().reset_index(name='value').sort_values(['bowler','mat_no'])
runs_ts3['mat_no']=runs_ts3['mat_no']+1
wickets_ts3 ['mat_no']=wickets_ts3 ['mat_no']+1
#%%
#%%
# Removing the previous match column. It was being used for persistence modellig method.
# runs_ts3['runs_prev_mat'] = runs_ts3.groupby('striker')['value'].shift()
# wickets_ts3['wicks_prev_mat'] = wickets_ts3.groupby('bowler')['value'].shift()
# runs_ts3.dropna(subset=['runs_prev_mat','value'],inplace=True)
# wickets_ts3.dropna(subset=['wicks_prev_mat','value'],inplace=True)
#Droppig where runs or wickets are nan. Which means the player did not play the match
runs_ts3.dropna(subset=['value'],inplace=True)
wickets_ts3.dropna(subset=['value'],inplace=True)
runs_ts3['mat_no2'] = runs_ts3.groupby('striker').cumcount()
wickets_ts3['mat_no2'] = wickets_ts3.groupby('bowler').cumcount()
runs_ts3['mat_no2']=runs_ts3['mat_no2']+1
wickets_ts3 ['mat_no2']=wickets_ts3 ['mat_no2']+1

#%%
runs_ts3['cumsum_runs']=runs_ts3.groupby(['striker'])['value'].cumsum()
wickets_ts3['cumsum_wickets']=wickets_ts3.groupby(['bowler'])['value'].cumsum()

runs_ts3['cumsum_avg']= runs_ts3['cumsum_runs']/runs_ts3['mat_no'] 

wickets_ts3['cumsum_avg']= wickets_ts3['cumsum_wickets']/wickets_ts3['mat_no']
#%%
runs_ts3['runs_next_mat'] = runs_ts3.groupby('striker')['value'].shift(-1)
wickets_ts3['wicks_next_mat'] = wickets_ts3.groupby('bowler')['value'].shift(-1)
#%%
runs_ts4= runs_ts3.dropna(subset=['runs_next_mat','value'],inplace=False)
wickets_ts4= wickets_ts3.dropna(subset=['wicks_next_mat','value'],inplace=False)

#%%
# X = runs_ts3[['mat_no','value','cumsum_avg','runs_next_mat']].values
# train_size = int(len(X) * 0.8)
# train, test = X[1:train_size], X[train_size:]
# train_X, train_y = train[:,0], train[:,1]
# test_X, test_y = test[:,0], test[:,1]
# #%% 
# # persistence model
# def model_persistence(x):
#  return x
 
# # walk-forward validation
# predictions = list()
# for x in test_X:
#  yhat = model_persistence(x)
#  predictions.append(yhat)
rmse_runs = mean_squared_error(runs_ts4['runs_next_mat'], runs_ts4['cumsum_avg'],squared=False)
print('Test RMSE for runs: %.3f' % rmse_runs)
rmse_wickets = mean_squared_error(wickets_ts4['wicks_next_mat'], wickets_ts4['cumsum_avg'],squared=False)
print('Test RMSE for wickets: %.3f' % rmse_wickets)

# pyplot.plot(train_y)
# pyplot.plot([None for i in train_y] + [x for x in test_y])
# pyplot.plot([None for i in train_y] + [x for x in predictions])
# pyplot.show()
#%%
idx=runs_ts3.groupby(['striker'])['mat_no2'].transform(max) == runs_ts3['mat_no2']
runs_ts5= runs_ts3[idx]
idx2=wickets_ts3.groupby(['bowler'])['mat_no2'].transform(max) == wickets_ts3['mat_no2']
wickets_ts5= wickets_ts3[idx2]
#%%
runs_ts6= pd.merge(runs_ts5,df_23wc_data[['striker','batting_team']].drop_duplicates(),how='left', on='striker')
wickets_ts6= pd.merge(wickets_ts5,df_23wc_data[['bowler','bowling_team']].drop_duplicates(),how='left', on='bowler')
#%%
wickets_ts6.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\best_bowler.pkl')
runs_ts6.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\best_batter.pkl')