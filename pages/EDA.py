import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#import src.soporteClean as sc
#import src.soporteImagenes as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Prediction', page_icon="📊")

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")
            
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
