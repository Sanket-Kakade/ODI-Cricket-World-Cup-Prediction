# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 00:22:59 2023

@author: sanket
"""

import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st
from PIL import Image  
# import streamlit.web.cli as stcli
from streamlit.web.cli import main
#%%
st.title("Top batter and bowler of the World Cup")
st.sidebar.header("Top Batter and Bowler prediction")

sf_teams_ls= ['India','South Africa','New Zealand','Australia']
def prediction(fin1,fin2): 
    final_teams_ls= [fin1,fin2]
    batter_df= pd.read_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\best_batter.pkl')
    bowler_df= pd.read_pickle('K:\\Sanket-datascience\\CWC_prediction\\Data\\prediction_related_data\\best_bowler.pkl')

    batter_df['playing_in_final']= np.where(batter_df['batting_team'].isin([fin1,fin2]),1,0)
    bowler_df['playing_in_final']= np.where(bowler_df['bowling_team'].isin([fin1,fin2]),1,0)
    batter_df['pred_total_runs']= np.round(batter_df['cumsum_runs']+batter_df['playing_in_final']*batter_df['cumsum_avg'],0)
    bowler_df['pred_total_wickets']= np.round(bowler_df['cumsum_wickets']+bowler_df['playing_in_final']*bowler_df['cumsum_avg'],0)
    
    top_batter= batter_df[batter_df['pred_total_runs']==batter_df['pred_total_runs'].max()]['striker'].values[0]
    top_bowler= bowler_df[bowler_df['pred_total_wickets']==bowler_df['pred_total_wickets'].max()]['bowler'].values[0]
    top_score= int(batter_df[batter_df['pred_total_runs']==batter_df['pred_total_runs'].max()]['pred_total_runs'].values[0])
    top_wickets= int(bowler_df[bowler_df['pred_total_wickets']==bowler_df['pred_total_wickets'].max()]['pred_total_wickets'].values[0])    


    return top_batter,top_bowler,top_score,top_wickets

fin1= st.selectbox("Select 1st finalist", ['India', "Australia", "New Zealand","South Africa"])
# sepal_width1 = st.text_input("Team 2", "Type Here") 
fin2= st.selectbox("Select 2nd finalist", ['India', "Australia", "New Zealand","South Africa"])
if st.button("Predict the top players"):  
    top_batter,top_bowler,top_score,top_wickets= prediction(fin1,fin2)  
    st.write('Highest run scorer of the tournament: ', str(top_batter), 'with ',str(top_score),'runs')  
    st.write('Highest wicket taker of the tournament: ', str(top_bowler), 'with ',str(top_wickets),'wickets')  

top_batter,top_bowler,top_score,top_wickets= prediction(fin1,fin2)
