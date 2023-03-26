import numpy as np
import pandas as pd
import pickle
import math
from scipy import stats

#display
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,8)

#normalización
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#estandarización 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#encoders
from sklearn.preprocessing import OneHotEncoder  
from sklearn.preprocessing import OrdinalEncoder

#metrics
from sklearn import metrics

# warnings
import warnings
warnings.filterwarnings('ignore')

def analisis_basico(dataframe):
    print("_________________________________\n")
    print (f"1_Estructura de los datos: {dataframe.shape}")
    display(dataframe.head(2))
    display(dataframe.info())
    print("_________________________________\n")
    print("2_Número de filas duplicadas:") 
    print(dataframe.duplicated().sum())
    print("_________________________________\n")
    display(pd.concat([dataframe.isnull().sum(), dataframe.dtypes], axis = 1).rename(columns = {0: "nulos", 1: "dtypes"}))
    print("_________________________________\n")
    print("3_Descripción de las variables tipo Numéricas:")
    display(dataframe.describe().T)
    print("_________________________________\n")
    print("4_Descripción de las variables tipo Categóricas:")
    display(dataframe.describe(include = "object").T)
    print("_________________________________\n")
    print("5_Distribución variables pairplot:")
    sns.pairplot(data=dataframe);
    print("_________________________________\n")

def regplot_numericas(dataframe, columnas_drop, variable_respuesta):
    df_numericas = dataframe.select_dtypes(include = np.number)
    columnas = df_numericas.drop(columnas_drop, axis = 1)
    fig, axes = plt.subplots(nrows=int(columnas.shape[1]/2), ncols=int(columnas.shape[1] / 3), figsize = (10 * columnas.shape[1] / 2,10 * columnas.shape[1] / 3))
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
    fig.tight_layout();

def chart_categoricas(df, variable_respuesta):
    df_cate = df.select_dtypes(include = 'object')
    print(df_cate.shape[1])
    fig, axes = plt.subplots(nrows=math.ceil(df_cate.shape[1]/2), ncols=math.ceil(df_cate.shape[1] / 2), figsize = (10 * df_cate.shape[1] / 2, 10 * df_cate.shape[1] / 2))
    axes = axes.flat
    for i, columns in enumerate(df_cate.columns):
        df_cat = df.groupby(columns)[variable_respuesta].median().reset_index()
        sns.barplot(data = df_cat, 
            x = columns, 
            y = variable_respuesta,
            ax = axes[i]
            )
    fig.tight_layout();

def chart_boxplot(dataframe):
    df_numericas = dataframe.select_dtypes(include = np.number).drop(['id'], axis = 1)

    fig, ax = plt.subplots(df_numericas.shape[1], 1, figsize=(12, 2.5 * df_numericas.shape[1]))

    for i in range(len(df_numericas.columns)):
        sns.boxplot(x=df_numericas.columns[i], data=df_numericas, ax=ax[i])
    plt.tight_layout()
    plt.show();

def detectar_outliers(lista_columnas, dataframe):

    dict_indices = {}

    for i in lista_columnas:
        Q1 = np.nanpercentile(dataframe[i], 25)
        Q3 = np.nanpercentile(dataframe[i], 75)

        IQR = Q3 - Q1

        outlier_step = IQR * 1.5
        outliers_value = dataframe[(dataframe[i] < Q1 - outlier_step) | (dataframe[i] > Q3 + outlier_step)]

        if outliers_value.shape[0] > 0:
            dict_indices[i] = outliers_value.index.tolist()
        else:
            #dict_indices[i] = 'sin outliers'
            pass
    return dict_indices

def normalizacion(df, variable_respuesta):
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
    fig.tight_layout();
    return df

def estandarizacion(dataframe, columnas, input):
    data = dataframe[columnas]
    if input == 'media':
        modelo = StandardScaler()
    elif input == 'mediana':
        modelo = RobustScaler()
    else:
        print("aprende a escribir")
    modelo.fit(data)
    X = modelo.transform(data)
    with open('../data/modelo/estandarizacion.pkl', 'wb') as s:
        pickle.dump(modelo, s)
    return X

def encoding(dataframe, columnas, input):
    if input == 'sin orden':
        modelo = OneHotEncoder()
        for columna in columnas:
            transformados = modelo.fit_transform(dataframe[[columna]])
            
            oh_df = pd.DataFrame(transformados.toarray(), columns = modelo.get_feature_names_out(), dtype = int)
            
            dataframe[oh_df.columns] = oh_df
            
            dataframe.drop(columna, axis = 1, inplace = True)
            
            with open(f'../data/modelo/encoding_{columna}.pkl', 'wb') as s:
                pickle.dump(modelo, s)
        
        return dataframe
    elif input == 'con orden':
        for columna in columnas:
            if dataframe[columna].dtype == 'float64' or dataframe[columna].dtype == 'int64':
                dataframe[columna] = dataframe[columna].astype(int).astype(str)
            else:
                pass
            orden = pd.DataFrame(dataframe[columna].value_counts()).reset_index().sort_values(by=columna)['index'].unique().tolist()
            modelo = OrdinalEncoder(categories = [orden], dtype = int)
            transformados = modelo.fit_transform(dataframe[[columna]])
            dataframe[columna] = transformados
            with open(f'../data/encoding_{columna}.pkl', 'wb') as s:
                pickle.dump(modelo, s)
        return dataframe
    else:
        print("aprende a escribir")

def metricas(y_test, y_train, y_test_pred, y_train_pred, tipo_modelo):
    
    resultados = {'MAE': [metrics.mean_absolute_error(y_test, y_test_pred), metrics.mean_absolute_error(y_train, y_train_pred)],
                'MSE': [metrics.mean_squared_error(y_test, y_test_pred), metrics.mean_squared_error(y_train, y_train_pred)],
                'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))],
                'R2':  [metrics.r2_score(y_test, y_test_pred), metrics.r2_score(y_train, y_train_pred)],
                 "set": ["test", "train"]}
    df = pd.DataFrame(resultados)
    df["modelo"] = tipo_modelo
    return df
