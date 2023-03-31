import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#import src.soporteClean as sc
#import src.soporteImagenes as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="centered", initial_sidebar_state="collapsed", page_title='Diamond Price Prediction_Kaggle Competition', page_icon="💎")

#tracks = sc.importDatasets()

st.markdown(f'# Diamonds | datamad1022')
st.image(f'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg/640px-Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg')
st.markdown("""
This is the third project of the Ironhack's Data Analysis Bootcamp.

The dataset is available [here](https://www.kaggle.com/competitions/diamonds-datamad1022/data). 
Also [here](https://www.kaggle.com/competitions/diamonds-datamad1022) is the link to the Kaggle competition.

The aim of this project is to develop a machine learning model pipeline to predict the price of diamonds given the dataset and submit it to the competition. 
So basically, the aim is to win the competition with the most accurate prediction!
""")

col1, col2 = st.columns(2)

with col1:
   st.markdown("""
[EDA](/EDA)
""")

with col2:
   st.markdown("""
[prediction](/prediction)
""")
            
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
