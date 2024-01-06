# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import os
import glob

#%%
icc_tournamet_winners_data= 'K:\\Sanket-datascience\\CWC_prediction\\Data\\icc_tournaments_winners_list.csv'
df_icc_winners= pd.read_csv(icc_tournamet_winners_data, encoding='unicode_escape')

all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\odis_male_csv\\"
all_cwc_data_loc= 'K:\\Sanket-datascience\\CWC_prediction\\Data\\icc_mens_cricket_world_cup_male_csv\\'

ls_odi_info_files= glob.glob(all_odi_data_loc+'*info.csv')
ls_cwc_info_files= glob.glob(all_cwc_data_loc+'*info.csv')
ls_odi_data_files= glob.glob(all_odi_data_loc+'*.csv')
ls_cwc_data_files= glob.glob(all_cwc_data_loc+'*.csv')

ls_odi_data_files= [i for i in ls_odi_data_files if i not in ls_odi_info_files]
ls_cwc_data_files= [i for i in ls_cwc_data_files if i not in ls_cwc_info_files]
ls_cwc_data_files= [i for i in ls_cwc_data_files if 'all_matches' not in i]

#%%

# df_info_single= pd.read_csv(ls_odi_info_files[1799],skiprows=2,names= ["type", "subtype", "field", "player","code"])
# df_info_single.iloc[0,1]= 'team1'
# df_info_single.iloc[1,1]= 'team2'

# df_info_single= df_info_single[df_info_single['field']!='people']

# df_info_single.drop('code',axis=1,inplace=True)

# team1_nm= df_info_single[df_info_single['subtype']=='team1']['field'].values[0]
# team2_nm= df_info_single[df_info_single['subtype']=='team2']['field'].values[0]
# team1_mapping= ['team1_p'+str(i) for i in range(1,12)]
# team2_mapping= ['team2_p'+str(i) for i in range(1,12)]


# df_info_single.loc[~(df_info_single['player'].isnull())&(df_info_single['field']==team1_nm),'subtype']= team1_mapping
# df_info_single.loc[~(df_info_single['player'].isnull())&(df_info_single['field']==team2_nm),'subtype']= team2_mapping

# df_info_single.loc[~(df_info_single['player'].isnull()),'field']=df_info_single.loc[~(df_info_single['player'].isnull()),'player']
# df_info_single.drop('player',axis=1,inplace=True)
#%%
team1_mapping= ['team1_p'+str(i) for i in range(1,12)]
team2_mapping= ['team2_p'+str(i) for i in range(1,12)]
#%%
df_odi_info= pd.DataFrame()
for i in ls_odi_info_files:
    print (i)
    df_info_single= pd.read_csv(i,skiprows=2,names= ["type", "subtype", "field", "player","code"])
    df_info_single= df_info_single[df_info_single['field']!='people']
    df_info_single= df_info_single[df_info_single['subtype']!='player']

    df_info_single.loc[df_info_single[df_info_single['subtype'].duplicated(keep=False)].index, 'subtype'] = df_info_single['subtype'] + '_' + df_info_single.groupby('subtype').cumcount().add(1).astype(str)

    df_info_single.drop('code',axis=1,inplace=True)
    df_info_single.drop(['player','type'],axis=1,inplace=True)
    df_info_single= df_info_single.append(pd.Series(['match_id', i.split('\\')[-1].split('_')[0]], index=df_info_single.columns), ignore_index=True)
    


#    df_info_single.loc[(df_info_single['subtype']=='umpire'),'subtype']= np.array(['umpire_1','umpire_2'])
    df_info_single=df_info_single.T
    df_info_single.columns = df_info_single.iloc[0]
    df_info_single= df_info_single[1:]

#### Below code commented to avoid getting the players info in the matches table
#    team1_nm= df_info_single[df_info_single['subtype']=='team1']['field'].values[0]
#    team2_nm= df_info_single[df_info_single['subtype']=='team2']['field'].values[0]

#    df_info_single.loc[~(df_info_single['player'].isnull())&(df_info_single['field']==team1_nm),'subtype']= team1_mapping
#    df_info_single.loc[~(df_info_single['player'].isnull())&(df_info_single['field']==team2_nm),'subtype']= team2_mapping
#    df_info_single.loc[~(df_info_single['player'].isnull()),'field']=df_info_single.loc[~(df_info_single['player'].isnull()),'player']
#    df_info_single.drop('player',axis=1,inplace=True)
    df_info_single.reset_index(inplace=True, drop=True)
    df_odi_info= pd.concat([df_info_single,df_odi_info],ignore_index=True)
#%%
cols_to_drop= ['players_1','players_2', 'players_3', 'players_4', 'players_5', 'players_6',
'players_7', 'players_8', 'players_9', 'players_10', 'players_11',
'players_12', 'players_13', 'players_14', 'players_15', 'players_16',
'players_17', 'players_18', 'players_19', 'players_20', 'players_21','players_22', 
'date_1','date_2','date_3','reserve_umpire_1','reserve_umpire_2','eliminator','gender','match_number','season',
'umpire_1', 'umpire_2','reserve_umpire', 'match_referee','tv_umpire','player_of_match','player_of_match_1','player_of_match_2','method']

df_odi_info['date']= np.where(df_odi_info['date'].isnull(), df_odi_info['date_1'], df_odi_info['date'])  
df_odi_info['reserve_umpire']= np.where(df_odi_info['reserve_umpire'].isnull(), df_odi_info['reserve_umpire_1'], df_odi_info['reserve_umpire'])  
df_odi_info['winner']= np.where(df_odi_info['winner'].isnull(), df_odi_info['eliminator'], df_odi_info['winner'])  
df_odi_info.drop(cols_to_drop,axis=1,inplace=True)
#%%
#df_odi_info[df_odi_info['city'].isnull()]['venue'].unique()
#venue_city_map= 
#%%
df_city=df_odi_info[['city']].drop_duplicates()
city_cty_map= pd.read_csv('K:\Sanket-datascience\CWC_prediction\simplemaps_worldcities_basicv1.76\worldcities.csv')
city_cty_map= city_cty_map[['city','country']].drop_duplicates()
city_cty_map_final= pd.merge(df_city,city_cty_map,how='left', left_on='city',right_on='city')
city_cty_map_final.to_csv('city_cty_map.csv')
#%%
city_cty_map_final= pd.read_csv('K:\Sanket-datascience\CWC_prediction\Data\city_cty_map_corrected.csv', index_col=0)
city_cty_map_final= city_cty_map_final[~city_cty_map_final['country_corrected'].isnull()]
city_cty_map_final= city_cty_map_final[['city','country_corrected']].drop_duplicates()

#%%
df_odi_info2= pd.merge(df_odi_info,city_cty_map_final,how='left', left_on='city',right_on='city')
#Dictionary of stadiums which have nulls against their City and hence country. Manually creating the country mapping for them. 
dict_stadium_cty_map= {'Pallekele International Cricket Stadium':'Sri Lanka',
'Rangiri Dambulla International Stadium':'Sri Lanka',
'Sharjah Cricket Stadium':'United Arab Emirates',
'Harare Sports Club':'Zimbabwe',
'Dubai International Cricket Stadium':'United Arab Emirates',
'Sydney Cricket Ground':'Australia',
'Melbourne Cricket Ground':'Australia',
'Queenstown Events Centre':'New Zealand',
'Adelaide Oval':'Australia',
'Sharjah Cricket Association Stadium':'United Arab Emirates',
'Mombasa Sports Club Ground':'Kenya',
'Dubai Sports City Cricket Stadium':'United Arab Emirates',
'Multan Cricket Stadium':'Pakistan',
'Sheikhupura Stadium':'Pakistan',
'Chittagong Divisional Stadium':'Bangladesh',
'Rawalpindi Cricket Stadium':'Pakistan',
'Perth Stadium':'Australia',
'Bulawayo Athletic Club':'Zimbabwe',
'Galle International Stadium':'Sri Lanka'} 


df_odi_info2['country_corrected'] = df_odi_info2['country_corrected'].fillna(df_odi_info2['venue'].map(dict_stadium_cty_map))
df_odi_info2.rename({'country_corrected':'host_cty'}, axis=1, inplace=True)
df_odi_info2.to_pickle('K:\Sanket-datascience\CWC_prediction\odi_results.pkl')
#%%
# df_icc_winners= df_icc_winners.drop(df_icc_winners[(df_icc_winners['Year']==2002)&(df_icc_winners['Tournament']=='Champions Trophy')].index)
# add_rows= {'Year':[2002,2002], 'Winner':['India','Sri Lanka'], 'Runner-up':['',''],'Tournament':['Champions Trophy','Champions Trophy']}
# df_add_rows= pd.DataFrame.from_dict(add_rows, orient='index').T
# df_icc_winners= df_icc_winners.append(df_add_rows, ignore_index = True)
# # df_icc_winners.groupby(['Year','Winner','Tournament']).count().reset_index()
# # df_icc_winners.pivot_table(index='Year',columns='Winner')
# # tmp_df['win_vs_top_eight_cum'] = tmp_df['win_vs_top_eight_flag'].shift().cumsum()

