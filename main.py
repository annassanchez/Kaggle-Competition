import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
#import src.soporteClean as sc
#import src.soporteImagenes as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Diamond Price Prediction_Kaggle Competition', page_icon="💎")

#tracks = sc.importDatasets()

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("""
    """)
with col2:
    st.markdown(f'# Diamonds | datamad1022')
    st.image(f'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg/640px-Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg')
    st.markdown("""
    This is the third project of the Ironhack's Data Analysis Bootcamp.

    The dataset is available [here](https://www.kaggle.com/competitions/diamonds-datamad1022/data). 
    Also [here](https://www.kaggle.com/competitions/diamonds-datamad1022) is the link to the Kaggle competition.

    The aim of this project is to develop a machine learning model pipeline to predict the price of diamonds given the dataset and submit it to the competition. 
    So basically, the aim is to win the competition with the most accurate prediction!

    ## Diamond Price

    The price of the diamonds is based on this 4 variables (known as the 4Cs):
    """)
with col3:
    st.markdown("""
    """)
col1, col2 = st.columns(2)
with col1:
    st.image('https://www.igi.org/assets/images/carat-sm.jpg')
with col2:
    st.markdown("""
    ### Carat
    Carat is the diamond weight. 
    One carat weighs 1/5 of a gram and is divided into 100 points, so a diamond weighing 1.07 ct. is referred to as "one carat and seven points."
    Diamonds of the same weight doesn't mean that the size appearance i the same is the same -> other factors as the `cut` or the material might affect also.""")

col1, col2 = st.columns(2)
with col1:
    st.image('https://www.igi.org/assets/images/color_graphic_sm.jpg')
with col2:
    st.markdown("""
    ### Color
    It refers to the color of the diamond. 
    The color is coded from `D` to `Z`, according to how clear it is. 
    To determine the correct color, all submitted diamonds are compared to an internationally accepted master set of stones, the colors of which range from `D`, or colorless (the most sought after) to `Z`, the most yellow/brown - aside from "fancy" yellow or brown.""")

col1, col2 = st.columns(2)
with col1:
    st.image('https://www.igi.org/assets/images/clarity-sm.jpg')
with col2:
    st.markdown("""
    ### Clarity
    It is a measurement of how clear the diamond is.
    There are two types of clarity characteristics: inclusions and blemishes. 
    In order to grade the clarity of a diamond, it is necessary to observe the number and nature of external and internal characteristics of the stone, as well as their size and position. 
    The difference is based on their locations: inclusions are enclosed within a diamond, while blemishes are external characteristics. IGI grading reports show plotted diagrams of clarity characteristics marked in red for internal and green for external features.
""")

col1, col2 = st.columns(2)
with col1:
    st.image('https://www.igi.org/assets/images/cut-sm.jpg')
with col2:
    st.markdown("""
    ### Cut
    It explains the quality of the diamond's cut. 
    The cut is measures the diamond's refeaction - evaluating how the light angle shifts through the diamond. 
    The better the cut, the diamond is clearer and shines brighter, therfore is more expensive. 
    If the cut is not well enough, the diamond will be more matt and less clear.
    """)

col1, col2, col3, col4 = st.columns([1,2,2,1])

with col1:
   st.markdown("""
""")

with col2:
   st.markdown("""
[EDA](/EDA)
""")

with col3:
   st.markdown("""
[prediction](/prediction)
""")

with col4:
   st.markdown("""
""")
            
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
