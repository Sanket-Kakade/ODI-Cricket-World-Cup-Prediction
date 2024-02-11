# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 00:51:36 2023

@author: sanket
"""

import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st
from PIL import Image  
# import streamlit.web.cli as stcli
from streamlit.web.cli import main
from sklearn.ensemble import VotingClassifier
import xgboost as xgb


#%%

# Load the trained model
pickle_in1 = open('.\Models\lr_model.pkl', 'rb')  
lr_model = pickle.load(pickle_in1)  
pickle_in2 = open('.\Models\svc_model.pkl', 'rb')  
svc_model = pickle.load(pickle_in2)  
pickle_in3 = open('.\Models\gbm_model.pkl', 'rb')  
gbm_model = pickle.load(pickle_in3)
pickle_in4 = open('.\\Models\\rf_model.pkl', 'rb')
rf = pickle.load(pickle_in4)

# pickle_in1 = open('K:\Sanket-datascience\CWC_prediction\Models\classifier1.pkl', 'rb')  
# classifier1 = pickle.load(pickle_in1) 
# Load the datasets
winners_pvt_feat= pd.read_pickle('.\Data\\prediction_related_data\\wc_winners.pkl')
winners_pvt_feat.drop('date',axis=1,inplace=True)
rup_pvt_feat= pd.read_pickle('.\Data\\prediction_related_data\\wc_rups.pkl')
rup_pvt_feat.drop('date',axis=1,inplace=True)
team_feats_feat= pd.read_pickle('.\Data\\prediction_related_data\\team_feats.pkl')
team_feats_feat.drop('date',axis=1,inplace=True)
head_to_head_feat= pd.read_pickle('.\Data\\prediction_related_data\\head2head.pkl')
head_to_head_feat.drop('date',axis=1,inplace=True)
team_wins_in_cty= pd.read_pickle('.\Data\\prediction_related_data\\win_pct_in_cty.pkl') 
team_wins_in_cty.drop('date',axis=1,inplace=True)
with open('.\Data\\prediction_related_data\\top_teams.pkl', 'rb') as f:
    top_eight_teams_ls= pickle.load(f)
with open('.\Data\\prediction_related_data\\wc_event.pkl', 'rb') as f:
    wc_event_ls= pickle.load(f)
with open('.\Data\\prediction_related_data\\col_seq.pkl', 'rb') as f:
    col_seq= pickle.load(f)
#%%
def get_binary_feats(input_data):
    conditions = [(input_data['toss_winner']==input_data['team_1'])&(input_data['toss_decision']=='bat'),
                  (input_data['toss_winner']==input_data['team_1'])&(input_data['toss_decision']=='field'),
                  (input_data['toss_winner']==input_data['team_2'])&(input_data['toss_decision']=='bat'),
                  (input_data['toss_winner']==input_data['team_2'])&(input_data['toss_decision']=='field')]
    choices = [input_data['team_1'], input_data['team_2'], input_data['team_2'],input_data['team_1'] ]
        
    input_data["bat_first_team"] = np.select(conditions, choices, default=np.nan)

    input_data['is_team1_home']= np.where(input_data['host_cty']==input_data['team_1'],1,0)
    input_data['is_team2_home']= np.where(input_data['host_cty']==input_data['team_2'],1,0)
    input_data['cwc_ct_flag']= np.where(input_data['event'].isin(wc_event_ls),1,0)
    input_data['team_1_top_eight_flag']= np.where(input_data['team_1'].isin(top_eight_teams_ls),1,0)
    input_data['team_2_top_eight_flag']= np.where(input_data['team_2'].isin(top_eight_teams_ls),1,0)
    input_data['toss_winner_team_1']= np.where(input_data['toss_winner']==input_data['team_1'],1,0)
    input_data['bat_first_flag_team_1']= np.where(input_data['bat_first_team']==input_data['team_1'],1,0)
    return input_data
def get_teams_features(ip_df):
    ip_df11= pd.merge(ip_df, team_feats_feat, left_on= ['team_1'], right_on=['team'], how='left')
    ip_df11.columns = ['team1_'+i if i in team_feats_feat else i for i in ip_df11.columns]
    ip_df1= pd.merge(ip_df11, team_feats_feat, left_on= ['team_2'], right_on=['team'], how='left' )
    ip_df1.columns = ['team2_'+i if i in team_feats_feat else i for i in ip_df1.columns]
    
    ip_df21= pd.merge(ip_df1,winners_pvt_feat, left_on= ['team_1'], right_on=['team'], how='left')
    ip_df21.rename({'no_of_wc_won':'team_1_no_of_wc_won'},axis=1,inplace=True)
    ip_df3= pd.merge(ip_df21,winners_pvt_feat, left_on= ['team_2'], right_on=['team'], how='left')
    ip_df3.rename({'no_of_wc_won':'team_2_no_of_wc_won'},axis=1,inplace=True)
    ip_df31= pd.merge(ip_df3, rup_pvt_feat, left_on= ['team_1'], right_on=['team'], how='left')
    ip_df31.rename({'no_of_wc_rup':'team_1_no_of_wc_rup'},axis=1,inplace=True)
    ip_df4= pd.merge(ip_df31, rup_pvt_feat, left_on= ['team_2'], right_on=['team'], how='left')
    ip_df4.rename({'no_of_wc_rup':'team_2_no_of_wc_rup'},axis=1,inplace=True)
    new_cols_created= ['team_1_no_of_wc_won','team_2_no_of_wc_won', 'team_1_no_of_wc_rup', 'team_2_no_of_wc_rup']
    ip_df4[new_cols_created]= ip_df4[new_cols_created].fillna(0)
    ip_df4.drop(['team_x','team_y'],axis=1,inplace=True)
    
    
    ip_df5= pd.merge(ip_df4,head_to_head_feat, left_on=['team_1', 'team_2'], right_on=['team_1','team_2'], how='left' )
    ip_df51= pd.merge(ip_df5,team_wins_in_cty, left_on=['team_1','host_cty'], right_on=['team','host_cty'],how='left' )
    ip_df51.rename({'win_ratio_in_cty':'team_1_win_ratio_in_cty'},axis=1,inplace=True)
    
    ip_df6= pd.merge(ip_df51,team_wins_in_cty, left_on=['team_2','host_cty'], right_on=['team','host_cty']  )
    ip_df6.rename({'win_ratio_in_cty':'team_2_win_ratio_in_cty'},axis=1,inplace=True)
    ip_df6.drop(['team_x','team_y'],axis=1,inplace=True)
    ip_df7= ip_df6[col_seq]
    return ip_df7
#%%

def prediction1(model_input):
    y_pred_lr= lr_model.predict_proba(model_input)
    y_pred_svc= svc_model.predict_proba(model_input)
    y_pred_gbm= gbm_model.predict_proba(model_input)
    y_pred_rf= rf.predict_proba(model_input)
    y_pred_ens= np.average([y_pred_lr,y_pred_svc,y_pred_gbm,y_pred_rf],axis=0)
    y_pred = np.argmax(y_pred_ens)
    pred_proba = np.max(y_pred_ens)    


    return y_pred,pred_proba
#%%

def main():  
    st.set_page_config(page_title="Winners Prediction", page_icon=":sharks:")
    st.sidebar.header("Winners Prediction")
    st.markdown("""
        <style>
        .heading {
        
            font-size:50px !important;
            text-align: center;
            font-weight: bold;
            
        }
        </style>
        """,unsafe_allow_html=True)

    st.markdown('''<p class="heading"> Winner Prediction </p>''', unsafe_allow_html=True)
    
    html_temp = """  
    <div style="background-color: #4684af; padding: 0px">  
    <h3 style="color: #000000; text-align: center;">Select the playing teams & other details</h2>  
    </div>  
    """  
  
    st.markdown(html_temp, unsafe_allow_html=True)  
    cols = st.columns(2)
    team_ls= ['India', "Australia", "New Zealand","South Africa","West Indies"]
    team_1= cols[0].selectbox("Select 1st team", team_ls)
    team_ls2=[i for i in team_ls if i!=team_1]
    team_2= cols[1].selectbox("Select 2nd team",team_ls2)
    cols = st.columns(2)
    host_cty= cols[0].selectbox("Select host nation", ['New Zealand', 'Australia', 'South Africa','Bangladesh','Sri Lanka','England', 'Pakistan', 'India'])
    event_nm= cols[1].selectbox("Select the event name", ["ICC Cricket World Cup", "Bilateral"])
    toss_winner = st.selectbox("Toss won by", [team_1,team_2])  
    toss_decision = st.selectbox("Toss decision",['Bat','Field'] ).lower() 
# team_1= 'India'
# team_2= 'New Zealand'
# toss_winner='New Zealand'
# toss_decision= 'Bat'.lower()
    ip_df= pd.DataFrame({'team_1':team_1,'team_2':team_2,'toss_winner':toss_winner,'toss_decision':toss_decision},index=[0])
    ip_df['host_cty']=host_cty
    ip_df['event']=event_nm
    ip_df['team_1_new'] = np.minimum(ip_df['team_1'], ip_df['team_2'])
    ip_df['team_2_new'] = np.maximum(ip_df['team_1'],ip_df['team_2'])
    ip_df['team_1']= ip_df['team_1_new']
    ip_df['team_2']= ip_df['team_2_new']
    ip_df.drop(['team_1_new','team_2_new'],axis=1,inplace=True)
    input_df= get_binary_feats(ip_df)
    input_df2= get_teams_features(input_df)
    team1= str(ip_df['team_1'].values[0])
    team2= str(ip_df['team_2'].values[0])
    if st.button("Predict the winner"):     
        result,pred_proba = prediction1(input_df2)
        result_str0 = f"""
            #### <span> The winner could be {team1}</span>
            """
        result_str1 = f"""
            #### <span> The winner could be {team2}</span>  
            """
        pred_prob = f"""
         ##### <span> Prediction Confidence {str(np.round(pred_proba*100))+'%'}</span>  
         """
        if result ==0:
            #st.write('The winner could be ', str(ip_df['team_1'].values[0]))  
            st.markdown(result_str0, unsafe_allow_html=True)
        else: 
            #st.write('The winner could be ', str(ip_df['team_2'].values[0]))
            st.markdown(result_str1, unsafe_allow_html=True)
        st.markdown(pred_prob, unsafe_allow_html=True)


if __name__ == '__main__':  
    main()  
#%%

#%%
