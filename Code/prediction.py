# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:28:32 2023

@author: sanket
"""

team_feats_df,winners_pvt2,rup_pvt2
raw_data_latest3['winner_encoded']= np.where(raw_data_latest3['winner']==raw_data_latest3['team_1'],'team_1','team_2')
raw_data_latest3['is_team1_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_1'],1,0)
raw_data_latest3['is_team2_home']= np.where(raw_data_latest3['host_cty']==raw_data_latest3['team_2'],1,0)
raw_data_latest3['cwc_ct_flag']= np.where(raw_data_latest3['event'].isin(wc_event_ls),1,0)
raw_data_latest3['team_1_top_eight_flag']= np.where(raw_data_latest3['team_1'].isin(top_eight_teams_ls),1,0)
raw_data_latest3['team_2_top_eight_flag']= np.where(raw_data_latest3['team_2'].isin(top_eight_teams_ls),1,0)
raw_data_latest3['toss_winner_team_1']= np.where(raw_data_latest3['toss_winner']==raw_data_latest3['team_1'],1,0)
raw_data_latest3['bat_first_flag_team_1']= np.where(raw_data_latest3['bat_first_team']==raw_data_latest3['team_1'],1,0)
head_to_head_win_loss_pct