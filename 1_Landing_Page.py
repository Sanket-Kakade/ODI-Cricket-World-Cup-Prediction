# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:06:57 2023

@author: sanket
"""

import streamlit as st
import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st
from PIL import Image  
# import streamlit.web.cli as stcli
from streamlit.web.cli import main
#%%

st.set_page_config(page_title="World Cup Prediction App",page_icon=":trophy:")

st.title("Welcome to World Cup prediction app! :trophy:")
title_alignment="""
<style>
Welcome to World Cup prediction app! :trophy:{
  text-align: center
}
</style>
"""
st.markdown(title_alignment, unsafe_allow_html=True)
st.markdown(
    """
This application predicts the winner of the ongoing ICC cricket World Cup 2023 being held in India. 
"""
)
st.markdown(
    """
The underneath AI model is built considering the matches till league stage of the tournament i.e. matches till 14th Nov are considered for prediction. 
"""
)

# image = Image.open('K:\Sanket-datascience\CWC_prediction\Code\captains_photo.jpg')
image = Image.open('.\Code\captains_photo.jpg')

left_co, cent_co,last_co = st.columns(3)
with left_co:
    st.image(image, caption=None, width=700, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

#%%
# # Load the trained model  
# pickle_in1 = open('K:\Sanket-datascience\CWC_prediction\Models\classifier1.pkl', 'rb')  
# classifier1 = pickle.load(pickle_in1)  

# def prediction1(ip_ls): 
#     random_ip= np.random.rand(1,43)
#     prediction = classifier1.predict(random_ip)  
#     proba= 0.51
#     return prediction,proba
# def main():  

#     st.set_page_config(page_title="Winners Prediction", page_icon=":sharks:")
#     st.sidebar.header("Winners Prediction")

#     st.title("CWC 23 Prediction")  

#     html_temp = """  
#     <div style="background-color: #4684af; padding: 16px">  
#     <h2 style="color: #000000; text-align: center;">Enter the inputs below</h2>  
#     </div>  
#     """  
  
#     st.markdown(html_temp, unsafe_allow_html=True)  
  
#     # sepal_length1 = st.text_input("Team 1", "Type Here") 
#     sepal_length1= st.selectbox("Select 1st team", ['India', "Australia", "NZ","SA"])
#     # sepal_width1 = st.text_input("Team 2", "Type Here") 
#     sepal_width1 = st.selectbox("Select 2nd team", ['India', "Australia", "NZ","SA"])  
    
#     petal_length1 = st.selectbox("Toss won by", [sepal_length1,sepal_width1])  
#     petal_width1 = st.selectbox("Toss decision",['Bat','Bowl'] )  
#     result = ""  

#     if st.button("Predict the winner"):  
#         result,pred_proba = prediction1([sepal_length1, sepal_width1, petal_length1, petal_width1])  
#         st.write('The winner could be ', str(result))  
#         st.write('Prediction confidence', str(pred_proba*100)+'%')
    
# if __name__ == '__main__':  
#     main()                  