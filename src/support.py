import numpy as np
import pandas as pd
import pickle

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
    with open('datos/estandarizacion.pkl', 'wb') as s:
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
            
            with open(f'datos/encoding_{columna}.pkl', 'wb') as s:
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
            with open(f'datos/encoding_{columna}.pkl', 'wb') as s:
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
