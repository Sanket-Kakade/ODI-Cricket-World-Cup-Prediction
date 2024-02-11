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
def change_label_style(label, font_size='12px', font_color='white', font_family='sans-serif'):
    html = f"""
    <script>
        var elems = window.parent.document.querySelectorAll('p');
        var elem = Array.from(elems).find(x => x.innerText == '{label}');
        elem.style.fontSize = '{font_size}';
        elem.style.color = '{font_color}';
        elem.style.fontFamily = '{font_family}';
    </script>
    """
    st.components.v1.html(html)
#%%
st.set_page_config(page_title="World Cup Prediction App",page_icon=":trophy:",layout='wide')

title_alignment="""
<style>
Welcome to World Cup prediction app! :trophy:{
  text-align: center
}
</style>
"""
st.markdown("""
<style>
.heading {

    font-size:50px !important;
    text-align: center;
    font-weight: bold;
    
}
.text{
      font-size:25px !important;
      text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('''<p class="heading"> Welcome to World Cup prediction app! &#127942 </p>''', unsafe_allow_html=True)
#st.markdown(title_alignment, unsafe_allow_html=True)
st.markdown(
    """
    <p class="text"> 
This application predicts the winner of the ICC cricket World Cup 2023 being held in India.
     </p>
""",unsafe_allow_html=True
)
st.markdown(
    """
    <p class="text"> 
The underneath AI model is built on the data from historinc matches since 2000.
    </p> 
""",unsafe_allow_html=True
)

image = Image.open('.\Code\captains_photo.jpg')

first_co, sec_co,third_col = st.columns([2, 5, 2])
with sec_co:
    st.image(image, caption=None, width=250 , use_column_width=True, clamp=False, channels="RGB", output_format="auto")

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