# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:36:49 2023

@author: sanket
"""


import pandas as pd
import numpy as np
import os
import glob
#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "odi_results.pkl"
raw_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)

raw_data_latest= raw_data[raw_data['date']>='2015/02/14'].reset_index(drop=True)
teams_ls= list(pd.unique(raw_data_latest[['team_1', 'team_2']].values.ravel('K')))
top_eight_teams_ls= ['South Africa','Australia','New Zealand','England','India','Pakistan','Sri Lanka','West Indies']
wc_event_ls= ['ICC Cricket World Cup', 'ICC World Cup']
#%%
# team_feats will have the features of each team for eg. team's win ratio, team's last 10 matches win ratio, team's win ratio at home etc. 
team_feats_df= pd.DataFrame()
for i in teams_ls:
    tmp_df= raw_data_latest.loc[np.where((raw_data_latest.loc[:,['team_1','team_2']]==i)==True)[0],:].sort_values('date')
    win_ratio= tmp_df[tmp_df['winner']==i].shape[0]/tmp_df.shape[0]
    
    wins_at_home= tmp_df[(tmp_df['winner']==i)&(tmp_df['host_cty']==i)].shape[0]
    mats_at_home= tmp_df[(tmp_df['host_cty']==i)].shape[0]
    win_ratio_in_home_cty= mats_at_home and wins_at_home/mats_at_home or 0  # a / b

    wins_vs_top_eight= tmp_df[(tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls)&(tmp_df['winner']==i))|(tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)&(tmp_df['winner']==i)].shape[0]
    mats_vs_top_eight= tmp_df[(tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls))|(tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)].shape[0]
    win_ratio_vs_top_eight= mats_vs_top_eight and wins_vs_top_eight/mats_vs_top_eight or 0  # a / b
    
    wins_in_wc= tmp_df[(tmp_df['winner']==i)&(tmp_df['event'].isin(wc_event_ls))].shape[0]
    mats_in_wc= tmp_df[(tmp_df['event'].isin(wc_event_ls))].shape[0]
    win_ratio_in_wc= mats_in_wc and wins_in_wc/mats_in_wc or 0  # a / b
    
    last_ten_mat= tmp_df.tail(10)
    wins_ratio_in_last_ten_mats= last_ten_mat[last_ten_mat['winner']==i].shape[0]/last_ten_mat.shape[0]
    team_feats_df= team_feats_df.append({'team':i,'win_ratio':win_ratio,'win_ratio_in_home_cty':win_ratio_in_home_cty,
                                         'win_ratio_vs_top_eight':win_ratio_vs_top_eight,
                                         'win_ratio_in_wc':win_ratio_in_wc,
                                         'wins_ratio_in_last_ten_mats':wins_ratio_in_last_ten_mats}, ignore_index=True)
    
#%%
# team_grid will create win ratios of two teams playing against each other
team_grid= pd.DataFrame(index=teams_ls, columns=teams_ls)
for i in teams_ls:
    for j in teams_ls:
        single_df= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['team_2']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['team_1']==j)]
        mats_won= single_df[single_df['winner']==i].shape[0]
        mats_played= single_df.shape[0]
        win_ratio_of_two_teams= mats_played and mats_won/mats_played or 0  # a / b
        team_grid.loc[i,j]= win_ratio_of_two_teams
team_win_loss_pct= team_grid.unstack().swaplevel().reset_index()
team_win_loss_pct.columns= ['team_1','team_2','team1_win_pct_over_team2']
#%%
team_host_cty_grid= pd.DataFrame()
for i in teams_ls:
    for j in raw_data_latest['host_cty'].unique():
        single_df2= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['host_cty']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['host_cty']==j)]
        mats_won= single_df2[single_df2['winner']==i].shape[0]
        mats_played= single_df2.shape[0]
        win_ratio_in_cty= mats_played and mats_won/mats_played or 0  # a / b
        team_host_cty_grid.loc[i,j]= win_ratio_in_cty
team_win_loss_pct_in_cty= team_host_cty_grid.unstack().swaplevel().reset_index()
team_win_loss_pct_in_cty.columns= ['team','host_cty','team_win_pct_in_cty']
#### TODO: Check if making the ratio as zero for non playing Cty-Team combination is a right choice. 
#%%
#Merging and renaming the columns.
raw_data_latest1= pd.merge(raw_data_latest, team_feats_df, left_on= 'team_1', right_on='team', how='left')
raw_data_latest1.columns = ['team1_'+i if i in team_feats_df else i for i in raw_data_latest1.columns]
raw_data_latest2= pd.merge(raw_data_latest1, team_feats_df, left_on= 'team_2', right_on='team', how='left' )
raw_data_latest2.columns = ['team2_'+i if i in team_feats_df else i for i in raw_data_latest2.columns]
#%%
#Removing the tied or no result matches
raw_data_latest3= raw_data_latest2[(raw_data_latest2['winner']==raw_data_latest2['team_1'])|(raw_data_latest2['winner']==raw_data_latest2['team_2'])]
raw_data_latest3['winner_encoded']= np.where(raw_data_latest3['winner']==raw_data_latest3['team_1'],'team_1','team_2')
raw_data_latest3['is_team1_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_1'],1,0)
raw_data_latest3['is_team2_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_2'],1,0)

#%%
raw_data_latest4= pd.merge(raw_data_latest3,team_win_loss_pct, left_on=['team_1', 'team_2'], right_on=['team_1','team_2']  )
raw_data_latest5= pd.merge(raw_data_latest4,team_win_loss_pct_in_cty, left_on=['team_1','host_cty'], right_on=['team','host_cty']  )

#%%
cols_to_drop = ['team_1', 'team_2', 'date', 'event', 'venue', 'city', 'toss_winner','toss_decision', 'winner', 'winner_wickets', 'match_id', 'winner_runs',
       'outcome', 'host_cty','team1_team','team2_team','team']
raw_data_latest5.drop(cols_to_drop,axis=1,inplace=True)
#%%
raw_data_latest5.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\feat_df.pkl')
