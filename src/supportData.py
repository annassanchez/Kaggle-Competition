import pickle
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math

def importDatasets():
    df = pd.read_csv('./data/train.csv')
    return df

def dropColumns(data, lista):
    if len(lista) == 0:
        return data
    else:
        return data.drop(lista, axis = 1)

def null_analysis(dataframe):
    return pd.concat([dataframe.isnull().sum(), dataframe.dtypes], axis = 1).rename(columns = {0: "nulos", 1: "dtypes"})

def normalizacion(df, variable_respuesta):
    """
    esta función normaliza la variable respuesta y da los resultados estadísticos (test de shapiro) para poder evaluar la respuesta
    parámetros: df -> dataframe a normalizar
        variable_respuesta -> la variable que se quiere normalizar
    output: dataframe con las variable respuesta y las normalizaciones:
        -logarítmica
        -raíz cuadrada
        -boxcox
    """
    df[f'{variable_respuesta}_LOG'] = df[variable_respuesta].apply(lambda x: np.log(x) if x != 0 else 0)
    df[f'{variable_respuesta}_SQRT'] = df[variable_respuesta].apply(lambda x: math.sqrt(x) if x != 0 else 0)
    df[f'{variable_respuesta}_BC'], lambda_ajustada = stats.boxcox(df[variable_respuesta])

    print('original', stats.shapiro(df[f"{variable_respuesta}"]), 
          '\n log:', stats.shapiro(df[f'{variable_respuesta}_LOG']), 
        '\n sqrt', stats.shapiro(df[f'{variable_respuesta}_SQRT']), 
        '\n bc:', stats.shapiro(df[f'{variable_respuesta}_BC']),
    )

    fig, axes = plt.subplots(1, 4, figsize = (20,5))

    axes[0].set_title(f'{variable_respuesta} original')
    axes[1].set_title(f'{variable_respuesta} logaritmica')
    axes[2].set_title(f'{variable_respuesta} SQRT')
    axes[3].set_title(f'{variable_respuesta} boxcox')

    sns.distplot(df[f'{variable_respuesta}'] ,ax = axes[0])
    sns.distplot(df[f'{variable_respuesta}_LOG'], ax = axes[1])
    sns.distplot(df[f'{variable_respuesta}_SQRT'], ax = axes[2])
    sns.distplot(df[f'{variable_respuesta}_BC'], ax = axes[3])
    fig.tight_layout()
    plt.savefig('./images/norm_transform.png');
    return  stats.shapiro(df[f"{variable_respuesta}"]), stats.shapiro(df[f'{variable_respuesta}_LOG']), stats.shapiro(df[f'{variable_respuesta}_SQRT']), stats.shapiro(df[f'{variable_respuesta}_BC']),