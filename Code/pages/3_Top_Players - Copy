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
    batter_df= pd.read_csv('K:\\Sanket-datascience\\CWC_prediction\\Data\\top_batters.csv')
    bowler_df= pd.read_csv('K:\Sanket-datascience\CWC_prediction\Data\\top_bowlers.csv')
    batter_df['runs_per_innings']= np.round(batter_df['Runs']/batter_df['Inns'])
    bowler_df['wick_per_innings']= np.round(bowler_df['Wkts']/batter_df['Inns'])
    batter_df['runs_aft_sf']= np.where(batter_df['Country'].isin(sf_teams_ls),batter_df['runs_per_innings']+batter_df['Runs'],batter_df['Runs'] )
    bowler_df['wicks_aft_sf']= np.where(bowler_df['Country'].isin(sf_teams_ls),bowler_df['wick_per_innings']+bowler_df['Wkts'],bowler_df['Wkts'] )
    batter_df['runs_aft_final']= np.where(batter_df['Country'].isin(final_teams_ls),2*batter_df['runs_per_innings']+batter_df['Runs'],batter_df['Runs'] )
    bowler_df['wicks_aft_final']= np.where(bowler_df['Country'].isin(final_teams_ls),2*bowler_df['wick_per_innings']+bowler_df['Wkts'],bowler_df['Wkts'] )

    top_batter= batter_df[batter_df['runs_aft_final']==batter_df['runs_aft_final'].max()]['Player'].values[0]
    top_bowler= bowler_df[bowler_df['wicks_aft_final']==bowler_df['wicks_aft_final'].max()]['Player'].values[0]
    return top_batter, top_bowler

fin1= st.selectbox("Select 1st finalist", ['India', "Australia", "New Zealand","South Africa"])
# sepal_width1 = st.text_input("Team 2", "Type Here") 
fin2= st.selectbox("Select 2nd finalist", ['India', "Australia", "New Zealand","South Africa"])
if st.button("Predict the top players"):  
    top_batter,top_bowler= prediction(fin1,fin2)  
    st.write('Highest run scorer of the tournament: ', str(top_batter))  
    st.write('Highest wicket taker of the tournament: ', str(top_bowler))  

top_batter,top_bowler= prediction(fin1,fin2)
