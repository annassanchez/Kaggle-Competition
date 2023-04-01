import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import src.supportData as sd
import src.supportImages as si
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title='EDA', page_icon="📊")

data = sd.importDatasets()

col1, col2= st.columns(2)

with col1:
   st.markdown("""
<a style='text-align: center' href="/" >main</a>
""", unsafe_allow_html=True)

with col2:
   st.markdown("""
<a style='text-align: center' href="/prediction">prediction</a>
""", unsafe_allow_html=True)
   
st.markdown("<h1 style='text-align: center'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)

st.markdown("## Preliminary analysis")
a,b,c,d, e, f = si.analisis_basico(data)
col1, col2 = st.columns([3,5])

with col1:
    st.markdown(f"""The train dataframe has size: `{a}`.
    The train dataset has `{e}` duplicated values, so there's no need to remove them.
    The train dataset has `{e}` null values, so there's no need to treat them.
    """)
    st.markdown(f"This are the categorical variables")
    st.dataframe(d)
    st.markdown(f"""
    The categorical values are: 
    - `cut`
    - `color`
    - `clarity`
    """)
    st.markdown(f"This are the numerical variables")
    st.dataframe(c)
    st.markdown(f"The answer variable is `price`")
    st.markdown(f"""
    Looking at the pairplot, we see that some of the fields seem to have no implication on the price at all - there seems to be no relation between the fields.
    These fields are:
    - `depth`
    - `table`
    
    The fields that seem to be related to the `price` value are:
    
    - `x`
    - `y`
    - `z`
    - `carat`
    
    We will continue to evaluate the relation between this columns and also the relation between the categorical columns and the answer variables.
    """)

with col2: 
    st.markdown(f"This pairplot shows how the numerical variables relate")
    st.image('./images/pairplot.png')

st.markdown("""## Numerical variables
### Answer variable
""")
            


col1, col2 = st.columns([3,2])    
with col1:  
    si.histogram_answer(data, 'price')
    st.image('./images/histogram_answer_.png')
with col2:
    st.markdown("""
    Here's the histogram of the answer variable. 

    As it is a numerical variable, we will try to see if it can be normalised to predict the values with a LinerRegression algorithm or instead some other models will be used for the prediction.

    What we see is that the variable is not normal - it looks like it is almost a bimodal distribution, so we will try to normalize it later, in the proeprocessing process.

    """)

st.markdown("""
    ### Predictor variables
    """)
col1, col2 = st.columns([3,1])    
with col1:  
    si.regplot_numericas(data, ['id', 'price'], 'price')
    st.image('./images/regplot_numericas.png')
with col2:  
    st.markdown("""
    As we see, the most correlated variables with `price` are:
    - `carat`
    - `x`
    - `y`
    - `z`

    We can drop `table` and `depth` because there seems to be no apparent relation.

    We will also ned to see the correlation matrix to see if these variables are related to each other.
    """)

col1, col2 = st.columns([3,2])

with col1:
    si.heatmap_numericas(data)
    st.image('./images/heatmap.png')
with col2:
    st.markdown("""
    Looking at the correlation, we can confirm that the variable that seem to weight more in orther to define the `price` variable are:

    - `x`
    - `y`
    - `z`
    - `carat`

    However, those variables seem also to have a big relation with one another. 
    After all, it makes sense that the `carat`, as it is the weight of the diamond, might also be related to its size.
    Even though they are related with one another, the correlation with the price is also pretty high, so I'll keep all of the columns in order to do the predictions.
    """)

st.markdown("""
    ## Categorical variables
    """)

col1, col2 = st.columns(2)

with col1:
   si.chart_categoricas_value(data, 'price')
   st.image('./images/chart_categoricas_value.png')
   st.markdown("""
   Categorical variables according to the answer variable (`price`).
   """)

with col2:
    si.chart_categoricas_count(data)
    st.image('./images/chart_categoricas_count.png')
    st.markdown("""
    Categorical variables count by types.
    """)
st.markdown("""
    As we see, even though to the price all of the categorical variables are quite similar, te count of the different classifications is not the same.

    Also, as we've seen, there's some importance according to the `cut`, `clarity` or `color` from the diamonds.
    """)
            
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
