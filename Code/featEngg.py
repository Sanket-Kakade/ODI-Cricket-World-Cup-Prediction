# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:36:49 2023

@author: sanket
"""


import pandas as pd
import numpy as np
import os
import glob
import datetime
import pickle
#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "odi_results.pkl"
past_winners_file= "icc_tournaments_winners_list.csv"
raw_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)
past_winners_df= pd.read_csv(all_odi_data_loc+past_winners_file)
raw_data['team_1_new'] = np.minimum(raw_data['team_1'], raw_data['team_2'])
raw_data['team_2_new'] = np.maximum(raw_data['team_1'], raw_data['team_2'])
raw_data['team_1']= raw_data['team_1_new']
raw_data['team_2']= raw_data['team_2_new']
raw_data.drop(['team_1_new','team_2_new'],axis=1,inplace=True)
raw_data_latest= raw_data.copy().sort_values('date')
teams_ls= sorted(list(pd.unique(raw_data_latest[['team_1', 'team_2']].values.ravel('K'))))
# top_eight_teams_ls= ['South Africa','Australia','New Zealand','England','India','Pakistan','Sri Lanka','West Indies']
top_eight_teams_ls= ['South Africa','Australia','New Zealand','England','India','Pakistan']
wc_event_ls= ['ICC Cricket World Cup', 'ICC World Cup']
#%%
# No of WC won in past by team
past_winners_df['date']=pd.to_datetime(past_winners_df['date'])
winners_pvt= pd.crosstab(past_winners_df['date'], past_winners_df['Winner'].fillna(0))
winners_pvt=winners_pvt.cumsum()
winners_pvt = winners_pvt.append(winners_pvt.tail(1)).reset_index()
winners_pvt.iloc[-1,0] = datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), '%Y-%m-%d %H:%M')
winners_pvt= winners_pvt.set_index('date')
winners_pvt2= winners_pvt.resample('1D').ffill()
for i in winners_pvt2.columns:
    winners_pvt2['test_col']= winners_pvt2[i] - winners_pvt2[i].shift()
    winners_pvt2[i]= np.where(winners_pvt2['test_col']==1, winners_pvt2[i]-1, winners_pvt2[i])
winners_pvt2.reset_index(inplace=True)
winners_pvt2['date'] = winners_pvt2['date'].dt.strftime('%Y/%m/%d')
winners_pvt2.drop('test_col',axis=1,inplace=True)
winners_pvt2= winners_pvt2.melt(id_vars=['date'],var_name="team",value_name="no_of_wc_won")

# No of WC runners up in past by team
past_winners_df['date']=pd.to_datetime(past_winners_df['date'])
rup_pvt= pd.crosstab(past_winners_df['date'], past_winners_df['Runner-up'].fillna(0))
rup_pvt=rup_pvt.cumsum()
rup_pvt = rup_pvt.append(rup_pvt.tail(1)).reset_index()
rup_pvt.iloc[-1,0] = datetime.datetime.strptime(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), '%Y-%m-%d %H:%M')
rup_pvt= rup_pvt.set_index('date')
rup_pvt2= winners_pvt.resample('1D').ffill()
for i in rup_pvt2.columns:
    rup_pvt2['test_col']= rup_pvt2[i] - rup_pvt2[i].shift()
    rup_pvt2[i]= np.where(rup_pvt2['test_col']==1, rup_pvt2[i]-1, rup_pvt2[i])
rup_pvt2.reset_index(inplace=True)
rup_pvt2['date'] = rup_pvt2['date'].dt.strftime('%Y/%m/%d')
rup_pvt2.drop('test_col',axis=1,inplace=True)
rup_pvt2= rup_pvt2.melt(id_vars=['date'],var_name="team",value_name="no_of_wc_rup")



# wc_win_data= pd.merge(raw_data_latest[['date']], winners_pvt2, left_on='date',right_on='Date', how='left')

#%%
# Deriving Batting first team
conditions = [(raw_data_latest['toss_winner']==raw_data_latest['team_1'])&(raw_data_latest['toss_decision']=='bat'),
              (raw_data_latest['toss_winner']==raw_data_latest['team_1'])&(raw_data_latest['toss_decision']=='field'),
              (raw_data_latest['toss_winner']==raw_data_latest['team_2'])&(raw_data_latest['toss_decision']=='bat'),
              (raw_data_latest['toss_winner']==raw_data_latest['team_2'])&(raw_data_latest['toss_decision']=='field')]
choices = [raw_data_latest['team_1'], raw_data_latest['team_2'], raw_data_latest['team_2'],raw_data_latest['team_1'] ]
    
raw_data_latest["bat_first_team"] = np.select(conditions, choices, default=np.nan)
#%%
# team_feats will have the features of each team for eg. team's win ratio, team's last 10 matches win ratio, team's win ratio at home etc. 
team_feats_df= pd.DataFrame()
cols_to_drop1= ['team_1','team_2','event', 'venue', 'city', 'toss_winner',
       'toss_decision', 'winner','winner_wickets', 'match_id', 'winner_runs',
       'outcome', 'host_cty', 'mat_flag', 'win_flag', 'win_cum', 'mat_cum','home_mat_flag', 
       'home_win_flag', 'home_win_cum','home_mat_cum',
       'toss_win_mat_flag','toss_win_mat_win_flag','toss_win_mat_cum','toss_win_mat_win_cum',
       'bat_first_mat_flag','bat_first_mat_win_flag','bat_first_mat_cum','bat_first_mat_win_cum',
       'bat_second_mat_flag','bat_second_mat_win_flag','bat_second_mat_cum','bat_second_mat_win_cum',
       'mat_vs_top_eight_flag','win_vs_top_eight_flag', 'win_vs_top_eight_cum', 'mat_vs_top_eight_cum',
        'mat_in_wc_flag', 'win_in_wc_flag','win_in_wc_cum', 'mat_in_wc_cum',
        'win_in_last_ten_mat','win_in_last_five_mat','win_in_last_three_mat','bat_first_team']
for i in teams_ls:
    tmp_df= raw_data_latest[(raw_data_latest['team_1']==i)|(raw_data_latest['team_2']==i)]
# Overall win ratio of a team
    tmp_df['mat_flag']= 1
    tmp_df['win_flag']= np.where(tmp_df['winner']==i,1,0)
    tmp_df['win_cum'] = tmp_df['win_flag'].shift().cumsum()
    tmp_df['mat_cum'] = tmp_df['mat_flag'].shift().cumsum()
    tmp_df['win_ratio']= tmp_df['win_cum']/tmp_df['mat_cum']

# Win ratio of a team at their home cty
    tmp_df['home_mat_flag']= np.where(tmp_df['host_cty']==i,1,0)
    tmp_df['home_win_flag']= np.where((tmp_df['host_cty']==i)&(tmp_df['winner']==i),1,0)
    tmp_df['home_win_cum'] = tmp_df['home_win_flag'].shift().cumsum()
    tmp_df['home_mat_cum'] = tmp_df['home_mat_flag'].shift().cumsum()
    tmp_df['win_ratio_home']= tmp_df['home_win_cum']/tmp_df['home_mat_cum']

# Win ratio of a team after winning the toss    
    tmp_df['toss_win_mat_flag']= np.where(tmp_df['toss_winner']==i,1,0)
    tmp_df['toss_win_mat_win_flag']= np.where((tmp_df['toss_winner']==i)&(tmp_df['winner']==i),1,0)
    tmp_df['toss_win_mat_cum'] = tmp_df['toss_win_mat_flag'].shift().cumsum()
    tmp_df['toss_win_mat_win_cum'] = tmp_df['toss_win_mat_win_flag'].shift().cumsum()
    tmp_df['win_ratio_toss_win']= tmp_df['toss_win_mat_win_cum']/tmp_df['toss_win_mat_cum']

# Win ratio of a team when batting first
    tmp_df['bat_first_mat_flag']= np.where(tmp_df['bat_first_team']==i,1,0)
    tmp_df['bat_first_mat_win_flag']= np.where((tmp_df['bat_first_team']==i)&(tmp_df['winner']==i),1,0)
    tmp_df['bat_first_mat_cum'] = tmp_df['bat_first_mat_flag'].shift().cumsum()
    tmp_df['bat_first_mat_win_cum'] = tmp_df['bat_first_mat_win_flag'].shift().cumsum()
    tmp_df['win_ratio_bat_first']= tmp_df['bat_first_mat_win_cum']/tmp_df['bat_first_mat_cum']

# Win ratio of a team when batting second
    tmp_df['bat_second_mat_flag']= np.where(tmp_df['bat_first_team']!=i,1,0)
    tmp_df['bat_second_mat_win_flag']= np.where((tmp_df['bat_first_team']!=i)&(tmp_df['winner']==i),1,0)
    tmp_df['bat_second_mat_cum'] = tmp_df['bat_second_mat_flag'].shift().cumsum()
    tmp_df['bat_second_mat_win_cum'] = tmp_df['bat_second_mat_win_flag'].shift().cumsum()
    tmp_df['win_ratio_bat_second']= tmp_df['bat_second_mat_win_cum']/tmp_df['bat_second_mat_cum']


# Win ratio of a team against top teams
    tmp_df['mat_vs_top_eight_flag']= np.where(((tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls)))|((tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)),1,0)
    tmp_df['win_vs_top_eight_flag']= np.where(((tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls))&(tmp_df['winner']==i))|((tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)&(tmp_df['winner']==i)),1,0)
    tmp_df['win_vs_top_eight_cum'] = tmp_df['win_vs_top_eight_flag'].shift().cumsum()
    tmp_df['mat_vs_top_eight_cum'] = tmp_df['mat_vs_top_eight_flag'].shift().cumsum()
    tmp_df['win_ratio_vs_top_eight']= tmp_df['win_vs_top_eight_cum']/tmp_df['mat_vs_top_eight_cum']
    
# Win ratio of a team in world cups
    tmp_df['mat_in_wc_flag']= np.where(tmp_df['event'].isin(wc_event_ls),1,0)
    tmp_df['win_in_wc_flag']= np.where(((tmp_df['winner']==i)&(tmp_df['event'].isin(wc_event_ls))),1,0)
    tmp_df['win_in_wc_cum'] = tmp_df['win_in_wc_flag'].shift().cumsum()
    tmp_df['mat_in_wc_cum'] = tmp_df['mat_in_wc_flag'].shift().cumsum()
    tmp_df['win_ratio_wc']= tmp_df['win_in_wc_cum']/tmp_df['mat_in_wc_cum']
 
# Win ratio of a team in last 5 and 10 matches
    tmp_df['win_in_last_ten_mat'] = tmp_df['win_flag'].rolling(min_periods=1, window=10, closed='left').sum()
    tmp_df['win_ratio_in_last_ten_mat'] = tmp_df['win_in_last_ten_mat']/10
    tmp_df['win_in_last_five_mat'] = tmp_df['win_flag'].rolling(min_periods=1, window=5, closed='left').sum()
    tmp_df['win_ratio_in_last_five_mat'] = tmp_df['win_in_last_five_mat']/5
    tmp_df['win_in_last_three_mat'] = tmp_df['win_flag'].rolling(min_periods=1, window=3, closed='left').sum()
    tmp_df['win_ratio_in_last_three_mat'] = tmp_df['win_in_last_three_mat']/3
    tmp_df['win_in_last_mat'] = tmp_df['win_flag'].rolling(min_periods=1, window=1, closed='left').sum()
    tmp_df['win_in_last_mat'] = tmp_df['win_in_last_mat']/1
    
    
    tmp_df['team']=i
    tmp_df.drop(cols_to_drop1, axis=1, inplace=True)
    team_feats_df= team_feats_df.append(tmp_df,ignore_index=True)
# tmp_df['win_in_wc_flag']= tmp_df['home_win_cum']/tmp_df['home_mat_cum']
#%%
# team_feats_df= pd.DataFrame()
# for i in teams_ls:
#     tmp_df= raw_data_latest.loc[np.where((raw_data_latest.loc[:,['team_1','team_2']]==i)==True)[0],:].sort_values('date')
#     win_ratio= tmp_df[tmp_df['winner']==i].shape[0]/tmp_df.shape[0]
    
#     wins_at_home= tmp_df[(tmp_df['winner']==i)&(tmp_df['host_cty']==i)].shape[0]
#     mats_at_home= tmp_df[(tmp_df['host_cty']==i)].shape[0]
#     win_ratio_in_home_cty= mats_at_home and wins_at_home/mats_at_home or 0  # a / b

#     wins_vs_top_eight= tmp_df[(tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls)&(tmp_df['winner']==i))|(tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)&(tmp_df['winner']==i)].shape[0]
#     mats_vs_top_eight= tmp_df[(tmp_df['team_1']==i)&(tmp_df['team_2'].isin(top_eight_teams_ls))|(tmp_df['team_1'].isin(top_eight_teams_ls))&(tmp_df['team_2']==i)].shape[0]
#     win_ratio_vs_top_eight= mats_vs_top_eight and wins_vs_top_eight/mats_vs_top_eight or 0  # a / b
    
#     wins_in_wc= tmp_df[(tmp_df['winner']==i)&(tmp_df['event'].isin(wc_event_ls))].shape[0]
#     mats_in_wc= tmp_df[(tmp_df['event'].isin(wc_event_ls))].shape[0]
#     win_ratio_in_wc= mats_in_wc and wins_in_wc/mats_in_wc or 0  # a / b
    
#     last_ten_mat= tmp_df.tail(10)
#     wins_ratio_in_last_ten_mats= last_ten_mat[last_ten_mat['winner']==i].shape[0]/last_ten_mat.shape[0]
#     team_feats_df= team_feats_df.append({'team':i,'win_ratio':win_ratio,'win_ratio_in_home_cty':win_ratio_in_home_cty,
#                                          'win_ratio_vs_top_eight':win_ratio_vs_top_eight,
#                                          'win_ratio_in_wc':win_ratio_in_wc,
#                                          'wins_ratio_in_last_ten_mats':wins_ratio_in_last_ten_mats}, ignore_index=True)
    
#%%
head_to_head_win_loss_pct= pd.DataFrame()
cols_to_drop2= ['winner','event','venue','city','toss_winner',
       'toss_decision','winner_wickets','match_id','winner_runs',
       'outcome','host_cty','mat_flag','team_1_win_flag','team_2_win_flag',
       'team_1_win_cum','team_2_win_cum','mat_cum','bat_first_team','team_1on2_win_last_five_mat_cum','team_2on1_win_last_five_mat_cum']
for i in teams_ls:
    for j in teams_ls:

        single_df= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['team_2']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['team_1']==j)]
        single_df['mat_flag']= 1
        single_df['team_1_win_flag']= np.where(single_df['winner']==i,1,0)
        single_df['team_2_win_flag']= np.where(single_df['winner']==j,1,0)


        single_df['team_1_win_cum'] = single_df['team_1_win_flag'].shift().cumsum()
        single_df['team_2_win_cum'] = single_df['team_2_win_flag'].shift().cumsum()
        
        single_df['mat_cum'] = single_df['mat_flag'].shift().cumsum()
        single_df['win_ratio_team_1on2']= single_df['team_1_win_cum']/single_df['mat_cum']
        single_df['win_ratio_team_2on1']= single_df['team_2_win_cum']/single_df['mat_cum']
        
# Win loss ratio in last 5/10 mats
        single_df['team_1on2_win_last_five_mat_cum'] = single_df['team_1_win_flag'].rolling(min_periods=1, window=5, closed='left').sum()
        single_df['win_ratio_team_1on2_last_five'] = single_df['team_1on2_win_last_five_mat_cum']/5
        single_df['team_2on1_win_last_five_mat_cum'] = single_df['team_2_win_flag'].rolling(min_periods=1, window=5, closed='left').sum()
        single_df['win_ratio_team_2on1_last_five'] = single_df['team_2on1_win_last_five_mat_cum']/5
        single_df['team_1on2_win_last_ten_mat_cum'] = single_df['team_1_win_flag'].rolling(min_periods=1, window=5, closed='left').sum()
        single_df['win_ratio_team_1on2_last_ten'] = single_df['team_1on2_win_last_ten_mat_cum']/10
        single_df['team_2on1_win_last_ten_mat_cum'] = single_df['team_2_win_flag'].rolling(min_periods=1, window=5, closed='left').sum()
        single_df['win_ratio_team_2on1_last_ten'] = single_df['team_2on1_win_last_ten_mat_cum']/10

        
        
        single_df.drop(cols_to_drop2, axis=1,inplace=True)
        head_to_head_win_loss_pct= head_to_head_win_loss_pct.append(single_df,ignore_index=True)
        
head_to_head_win_loss_pct.drop_duplicates(subset=['team_1','team_2','date'], inplace=True)
#%%

# # team_grid will create win ratios of two teams playing against each other
# team_grid= pd.DataFrame(index=teams_ls, columns=teams_ls)
# for i in teams_ls:
#     for j in teams_ls:
#         single_df= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['team_2']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['team_1']==j)]
#         mats_won= single_df[single_df['winner']==i].shape[0]
#         mats_played= single_df.shape[0]
#         win_ratio_of_two_teams= mats_played and mats_won/mats_played or 0  # a / b
#         team_grid.loc[i,j]= win_ratio_of_two_teams
# team_win_loss_pct= team_grid.unstack().swaplevel().reset_index()
# team_win_loss_pct.columns= ['team_1','team_2','team1_win_pct_over_team2']
#%%
cols_to_drop3= ['team_1','team_2','event','venue','city','toss_winner',
       'toss_decision','winner','winner_wickets','match_id','winner_runs',
       'outcome','mat_flag','team_win_flag','team_win_cum','mat_cum','bat_first_team']
team_win_loss_pct_in_cty= pd.DataFrame()
for i in teams_ls:
    for j in raw_data_latest['host_cty'].unique():
        single_df2= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['host_cty']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['host_cty']==j)]
        single_df2['mat_flag']= 1
        single_df2['team_win_flag']= np.where(single_df2['winner']==i,1,0)
        single_df2['team_win_cum'] = single_df2['team_win_flag'].shift().cumsum()
        single_df2['mat_cum'] = single_df2['mat_flag'].shift().cumsum()
        single_df2['team']=i
        single_df2['win_ratio_in_cty']= single_df2['team_win_cum']/single_df2['mat_cum']
        single_df2.drop(cols_to_drop3, axis=1,inplace=True)
        team_win_loss_pct_in_cty= team_win_loss_pct_in_cty.append(single_df2,ignore_index=True)

#%%
# team_host_cty_grid= pd.DataFrame()
# for i in teams_ls:
#     for j in raw_data_latest['host_cty'].unique():
        
#         single_df2= raw_data_latest[(raw_data_latest['team_1']==i)&(raw_data_latest['host_cty']==j)|(raw_data_latest['team_2']==i)&(raw_data_latest['host_cty']==j)]
#         mats_won= single_df2[single_df2['winner']==i].shape[0]
#         mats_played= single_df2.shape[0]
#         win_ratio_in_cty= mats_played and mats_won/mats_played or 0  # a / b
#         team_host_cty_grid.loc[i,j]= win_ratio_in_cty
# team_win_loss_pct_in_cty= team_host_cty_grid.unstack().swaplevel().reset_index()
# team_win_loss_pct_in_cty.columns= ['team','host_cty','team_win_pct_in_cty']
#### TODO: Toss winning impact on winning the match. Win loss ratio.
#### TODO: bat first, bat second record
#%%
#%%
#Merging and renaming the columns.
raw_data_latest1= pd.merge(raw_data_latest, team_feats_df, left_on= ['team_1','date'], right_on=['team','date'], how='left')
raw_data_latest1.columns = ['team1_'+i if i in team_feats_df else i for i in raw_data_latest1.columns]
raw_data_latest1.rename({'team1_date':'date'},axis=1,inplace=True)
raw_data_latest2= pd.merge(raw_data_latest1, team_feats_df, left_on= ['team_2','date'], right_on=['team','date'], how='left' )
raw_data_latest2.columns = ['team2_'+i if i in team_feats_df else i for i in raw_data_latest2.columns]
raw_data_latest2.rename({'team2_date':'date'},axis=1,inplace=True)
#%%
#Code to merge the winners and runners up data with the feature df. 
raw_data_latest31= pd.merge(raw_data_latest2, winners_pvt2, left_on= ['team_1','date'], right_on=['team','date'], how='left' )
raw_data_latest31.rename({'no_of_wc_won':'team_1_no_of_wc_won'},axis=1,inplace=True)
raw_data_latest32= pd.merge(raw_data_latest31, winners_pvt2, left_on= ['team_2','date'], right_on=['team','date'], how='left' )
raw_data_latest32.rename({'no_of_wc_won':'team_2_no_of_wc_won'},axis=1,inplace=True)

raw_data_latest33= pd.merge(raw_data_latest32, rup_pvt2, left_on= ['team_1','date'], right_on=['team','date'], how='left')
raw_data_latest33.rename({'no_of_wc_rup':'team_1_no_of_wc_rup'},axis=1,inplace=True)
raw_data_latest34= pd.merge(raw_data_latest33, rup_pvt2, left_on= ['team_2','date'], right_on=['team','date'], how='left')
raw_data_latest34.rename({'no_of_wc_rup':'team_2_no_of_wc_rup'},axis=1,inplace=True)

new_cols_created= ['team_1_no_of_wc_won','team_2_no_of_wc_won', 'team_1_no_of_wc_rup', 'team_2_no_of_wc_rup']
raw_data_latest34[new_cols_created]= raw_data_latest34[new_cols_created].fillna(0)
raw_data_latest34.drop(['team_x','team_y'],axis=1,inplace=True)
#%%
#Removing the tied or no result matches
raw_data_latest3= raw_data_latest34[(raw_data_latest34['winner']==raw_data_latest34['team_1'])|(raw_data_latest34['winner']==raw_data_latest34['team_2'])]
#%%
#Creating binary flags based on the available data and conditions
raw_data_latest3['winner_encoded']= np.where(raw_data_latest3['winner']==raw_data_latest3['team_1'],'team_1','team_2')
raw_data_latest3['is_team1_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_1'],1,0)
raw_data_latest3['is_team2_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_2'],1,0)
raw_data_latest3['cwc_ct_flag']= np.where(raw_data_latest3['event'].isin(wc_event_ls),1,0)
raw_data_latest3['team_1_top_eight_flag']= np.where(raw_data_latest3['team_1'].isin(top_eight_teams_ls),1,0)
raw_data_latest3['team_2_top_eight_flag']= np.where(raw_data_latest3['team_2'].isin(top_eight_teams_ls),1,0)
raw_data_latest3['toss_winner_team_1']= np.where(raw_data_latest3['toss_winner']==raw_data_latest3['team_1'],1,0)
raw_data_latest3['bat_first_flag_team_1']= np.where(raw_data_latest3['bat_first_team']==raw_data_latest3['team_1'],1,0)


#%%
raw_data_latest4= pd.merge(raw_data_latest3,head_to_head_win_loss_pct, left_on=['team_1', 'team_2','date'], right_on=['team_1','team_2','date']  )
raw_data_latest41= pd.merge(raw_data_latest4,team_win_loss_pct_in_cty, left_on=['team_1','host_cty','date'], right_on=['team','host_cty','date']  )
raw_data_latest41.rename({'win_ratio_in_cty':'team_1_win_ratio_in_cty'},axis=1,inplace=True)

raw_data_latest5= pd.merge(raw_data_latest41,team_win_loss_pct_in_cty, left_on=['team_2','host_cty','date'], right_on=['team','host_cty','date']  )
raw_data_latest5.rename({'win_ratio_in_cty':'team_2_win_ratio_in_cty'},axis=1,inplace=True)
raw_data_latest5.drop(['team_x','team_y'],axis=1,inplace=True)

#%%
non_nation_cty= ['Africa XI','Asia XI','ICC World XI']
wc23_cty= ['India','Sri Lanka','Australia','England','Pakistan','West Indies','South Africa','New Zealand ', 
'Bangladesh','Zimbabwe','Ireland','Afghanistan','Scotland','United Arab Emirates',
'Netherlands','Nepal','Papua New Guinea','United States of America']
raw_data_latest51= raw_data_latest5[~(raw_data_latest5['team1_team'].isin(non_nation_cty))&~(raw_data_latest5['team2_team'].isin(non_nation_cty))]
raw_data_latest6= raw_data_latest51[(raw_data_latest51['team1_team'].isin(wc23_cty))&(raw_data_latest51['team2_team'].isin(wc23_cty))]
# raw_data_latest6= raw_data_latest51.copy()
#%%
raw_data_latest7 = pd.get_dummies(raw_data_latest6, columns=['team_1','team_2' ])

#%%

# cols_to_drop = ['team_1', 'team_2', 'event', 'venue', 'city', 'toss_winner','toss_decision', 'winner', 'winner_wickets', 'match_id', 'winner_runs',
#        'outcome', 'host_cty','team1_team','team2_team','team']
cols_to_drop = ['event', 'venue', 'city', 'toss_winner','toss_decision', 'winner', 'winner_wickets', 'match_id', 'winner_runs',
       'outcome', 'host_cty','bat_first_team','team1_team','team2_team']

raw_data_latest8= raw_data_latest7.drop(cols_to_drop,axis=1,inplace=False)
#%%
winners_pvt_feat =winners_pvt2.groupby(['team'],as_index=False).last()
rup_pvt_feat= rup_pvt2.groupby(['team'],as_index=False).last()
team_feats_feat= team_feats_df.groupby(['team'],as_index=False).last()
head_to_head_feat= head_to_head_win_loss_pct.groupby(['team_1','team_2'],as_index=False).last()
team_wins_in_cty= team_win_loss_pct_in_cty.groupby(['host_cty','team'],as_index=False).last()
#%%
raw_data_latest8.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\feat_df_more_mts.pkl')
#%%

winners_pvt_feat.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\wc_winners.pkl')
rup_pvt_feat.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\wc_rups.pkl')
team_feats_feat.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\team_feats.pkl')
head_to_head_feat.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\head2head.pkl')
team_wins_in_cty.to_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\win_pct_in_cty.pkl')
with open('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\top_teams.pkl', 'wb') as f:
   pickle.dump(top_eight_teams_ls, f)
with open('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\wc_event.pkl', 'wb') as f:
    pickle.dump(wc_event_ls, f)
