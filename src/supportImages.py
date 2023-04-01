import numpy as np
import pandas as pd
import pickle
import math
from scipy import stats
from datetime import datetime

#display
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,8)
from IPython.display import clear_output

def analisis_basico(dataframe):
    """
    esta función coge un dataframe y saca los principales elementos preiminares de un eda: la estructura de datos, si hay nulos, duplicados, el tipo de variables numéricas o categórcias y un pairplot para ver la relación entre variables.
    arg: dataframe
    outpit: dataframe.shape
        test -> a dataframe that gives the columns of the given dataframe and if theis columns have null values
    """
    test = pd.concat([dataframe.isnull().sum(), dataframe.dtypes], axis = 1).rename(columns = {0: "nulos", 1: "dtypes"})
    sns.pairplot(data=dataframe.drop(['id'], axis = 1));
    plt.savefig('./images/pairplot.png')
    return dataframe.shape, test, dataframe.describe().T, dataframe.describe(include = "object").T, dataframe.duplicated().sum(), dataframe.isnull().sum()

def histogram_answer(df, answer_variable):
    sns.kdeplot(data = df, x = df[answer_variable])
    plt.savefig('./images/histogram_answer.png');

def regplot_numericas(dataframe, columnas_drop, variable_respuesta):
    """
    esta función da un chart que relaciona las columnas numéricas de un dataframe con la variable
    param: dataframe -> el dataframe a representar
        columnas_drop -> las columnas a borrar (un id alguna columna que no se quiera representar) -> se pasa en formato lista
        variable_respuesta -> las columnas a borrar (en este caso, la variable respuesta)
    """
    df_numericas = dataframe.select_dtypes(include = np.number)
    columnas = df_numericas.drop(columnas_drop, axis = 1)
    fig, axes = plt.subplots(nrows=int(columnas.shape[1]/3), ncols=int(columnas.shape[1]/2), figsize = (5 * columnas.shape[1] / 2, 5 * columnas.shape[1] / 3))
    axes = axes.flat
    for i, columns in enumerate(columnas.columns):
        sns.regplot(data = dataframe, 
            x = columns, 
            y = variable_respuesta, 
            ax = axes[i],
            color = 'gray',
            scatter_kws = {"alpha": 0.4}, 
            line_kws = {"color": "red", "alpha": 0.7 }
            )
    fig.tight_layout()
    plt.savefig('./images/regplot_numericas.png');

def heatmap_numericas(df):
    sns.heatmap(df.corr(), 
            cmap = "YlGnBu",
            annot = True)
    plt.savefig('./images/heatmap.png');

def chart_categoricas_count(df):
    """
    esta función toma un dataframe y presnta unos histogramas con las variables categóricas
    param: dataframe -> dataframe del que se sacan los gráficos
    """
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (30, 10))

    axes = axes.flat

    df_cat = df.select_dtypes(include = 'object')#.columns

    for i, colum in enumerate(df_cat.columns):
        chart = sns.countplot(
                x = df_cat[colum],
                #hue = df_cat['Offer_Accepted'],
                ax = axes[i])
        total = float(len(df_cat[colum]))
        for p in chart.patches:
            height = p.get_height()
            chart.text(p.get_x() + p.get_width() / 2., height + 3,
                    '{:.2f}%'.format((height / total) * 100),
                    ha='center')
    fig.tight_layout();
    plt.savefig('./images/chart_categoricas_count.png')

def chart_categoricas_value(df, variable_respuesta):
    """
    esta función es para hacer un chart que relacione las variables categóricas con la mediana del valor de la variable respuesta
    param: dataframe -> dataframe que se quiere representar
        variable_respuesta -> la variable respuesta del dataframe
    """
    df_cate = df.select_dtypes(include = 'object')
    fig, axes = plt.subplots(nrows=1, ncols=math.ceil(df_cate.shape[1]), figsize = (30, 10))
    axes = axes.flat
    for i, columns in enumerate(df_cate.columns):
        df_cat = df.groupby(columns)[variable_respuesta].median().reset_index()
        sns.barplot(data = df_cat, 
            x = columns, 
            y = variable_respuesta,
            ax = axes[i]
            )
    fig.tight_layout()
    plt.savefig('./images/chart_categoricas_value.png');

def chart_boxplot(dataframe):
    """
    esta funcion saca los boxplots de las variables numéricas - incluyendo la variable respuesta
    param: dataframe
    """
    df_numericas = dataframe.select_dtypes(include = np.number)

    fig, ax = plt.subplots(df_numericas.shape[1], 1, figsize=(12, 2.5 * df_numericas.shape[1]))

    for i in range(len(df_numericas.columns)):
        sns.boxplot(x=df_numericas.columns[i], data=df_numericas, ax=ax[i])
    plt.tight_layout()
    plt.show()
    plt.savefig('./images/chart_boxplot.png');