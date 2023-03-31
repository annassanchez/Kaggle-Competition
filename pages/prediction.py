import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#import src.soporteClean as sc
#import src.soporteImagenes as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Prediction', page_icon="🔮")

#tracks = sc.importDatasets()

st.markdown(f'# Diamonds | datamad1022')

col1, col2 = st.columns(2)

with col1:
   st.markdown("""
[EDA](/EDA)
[test](../data/train.csv)
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
