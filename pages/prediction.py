import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import src.supportData as sd
import src.supportImages as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='Prediction', page_icon="🔮")

#tracks = sc.importDatasets()

data = sd.importDatasets()

col1, col2= st.columns(2)

with col1:
   st.markdown("""
<a style='text-align: center' href="/" >main</a>
""", unsafe_allow_html=True)

with col2:
   st.markdown("""
<a style='text-align: center' href="/EDA">EDA</a>
""", unsafe_allow_html=True)
   
st.markdown("<h1 style='text-align: center'>Prediction</h1>", unsafe_allow_html=True)

col1, col2= st.columns(2, gap = 'medium')

with col1:
   st.markdown("""
<h2> Previous step</h2>
<div>Choose the columns you want to drop:</div>
""", unsafe_allow_html=True)

   options = st.multiselect(
      'Columns to drop',
      ['id', 'carat', 'depth', 'table', 'x', 'y', 'z', 'cut', 'color', 'clarity'],
      ['id', 'depth', 'table'])
   new_data = sd.dropColumns(data, options)
   st.dataframe(new_data, height=15)
   st.markdown('For the best result, I used all the columns except `id`, `table`, `depth`')
   st.markdown("""
<h2>Null management</h2>
<div>Let's see if we have null values:</div>
""", unsafe_allow_html=True)
   st.dataframe(sd.null_analysis(new_data))
   st.markdown("There are no nulls, so we can continue with the analysis")

with col2:
   st.markdown("""
<h2> Outliers </h2>
For the outliers, let's visualize them with a boxplot.
""", unsafe_allow_html=True)
   si.chart_boxplot(new_data)
   st.image('./images/chart_boxplot.png')
   st.markdown('''There are outliers on the following columns:
   - `carat`
   - `x`
   - `y`
   - `z`

   I decided to mantain them, in order not to distord the given dataset.
   ''')

col1, col2= st.columns(2, gap = 'medium')

with col1:
   st.markdown("""
<h2> Normalization</h2>
<div>We will try to normalise the data:</div>
""", unsafe_allow_html=True)
   st.image('./images/histogram_answer_.png')
   st.markdown("""
<div>Let's see if the variable can be normalised:</div>
""", unsafe_allow_html=True)
   sd.normalizacion(new_data, 'price')
   st.image('./images/norm_transform.png')
   st.write(sd.normalizacion(new_data, 'price'))
   st.markdown("""
<div>The p-value is smaller than 0.5 so the data cannot be normalised. That means that there's no point to do a LinearRegression</div>
""", unsafe_allow_html=True)
   
with col2:
   st.markdown("""
<h2> Standarization </h2>
As I didn't treat the outlier values, the recomendend scaler for this is the RobustScaler. 
However, feel free to choose another scaler for the transformation.
""", unsafe_allow_html=True)
   scaler = st.selectbox('Choose the scaler', ['RobustScaler', 'StandardScaler'])
   #sd.normalizacion(new_data, 'price')
   st.markdown('''There are outliers on the following columns:
   - `carat`
   - `x`
   - `y`
   - `z`
   I decided to mantain them, in order not to distord the given dataset.
   ''')
            
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
