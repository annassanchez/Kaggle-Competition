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

#modelo
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def analisis_basico(dataframe):
    """
    esta función coge un dataframe y saca los principales elementos preiminares de un eda: la estructura de datos, si hay nulos, duplicados, el tipo de variables numéricas o categórcias y un pairplot para ver la relación entre variables.
    param: dataframe
    """
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
    """
    esta función da un chart que relaciona las columnas numéricas de un dataframe con la variable
    param: dataframe -> el dataframe a representar
        columnas_drop -> las columnas a borrar (un id alguna columna que no se quiera representar) -> se pasa en formato lista
        variable_respuesta -> las columnas a borrar (en este caso, la variable respuesta)
    """
    print(f'distribución de las variables numéricas en relación con la variable respuesta: {variable_respuesta}')
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

def chart_categoricas_count(df):
    """
    esta función toma un dataframe y presnta unos histogramas con las variables categóricas
    param: dataframe -> dataframe del que se sacan los gráficos
    """
    print(f'este chart da la distribución de las variables categóricas')
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

def chart_categoricas_value(df, variable_respuesta):
    """
    esta función es para hacer un chart que relacione las variables categóricas con la mediana del valor de la variable respuesta
    param: dataframe -> dataframe que se quiere representar
        variable_respuesta -> la variable respuesta del dataframe
    """
    df_cate = df.select_dtypes(include = 'object')
    print(f'este chart da la relación de las variables categóricas con la variable respuesta: {variable_respuesta}')
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
    """
    esta funcion saca los boxplots de las variables numéricas - incluyendo la variable respuesta
    param: dataframe
    """
    print('detección de outliers')
    df_numericas = dataframe.select_dtypes(include = np.number).drop(['id'], axis = 1)

    fig, ax = plt.subplots(df_numericas.shape[1], 1, figsize=(12, 2.5 * df_numericas.shape[1]))

    for i in range(len(df_numericas.columns)):
        sns.boxplot(x=df_numericas.columns[i], data=df_numericas, ax=ax[i])
    plt.tight_layout()
    plt.show();

def detectar_outliers(lista_columnas, dataframe):
    """
    función que devuelve un diccionario con los índices del dataframe que contienen outliers para las columnas dadas.
    param: lista_columnas -> columnas numéricas de las que se quieren sacar los outliers
        dataframe -> el dataframe del que se quieren obtener los outliers
    output: dict_indices -> un diccionario con las columnas y los índices que contienen outliers del data frame
    """
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
    fig.tight_layout();
    return df

def estandarizacion(dataframe, columnas, input):
    """
    esta función hace la estandarización de las variables numéricas
    parámetros: df-> dataframe a estandarizar
        columnas -> la lista de las columnas numéricas que se quieren estandarizar
        input -> el tipo de estandarización que se quiere hacer:
            - media: StandardScaler()
            - mediana: RobustScaler()
    output: devuelve un dataframe con las columnas transformadas
    """
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
    """
    esta función hace un encoding con orden o sin orden de las columnas categóricas del dataframe
    parámetros: dataframe -> el dataframe de origen del que se quiere hacer el encoding
        columnas -> las columnas categóricas de las que se quiere hacer el encoding
        input -> texto que puede ser de dos tipos (según el encoding que se quiera hacer)
            - 'sin orden' -> OneHotEncoder
            - 'con orden' -> OrdinalEncoder
    output: el dataframe con las columnas encodeadas
    """
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
            with open(f'../data/modelo/encoding_{columna}.pkl', 'wb') as s:
                pickle.dump(modelo, s)
        return dataframe
    else:
        print("aprende a escribir")

def ordinal_map(df, columna, orden_valores):
    """
    esta función hace un encoding con un orden dado a partir de un mapeo
    parámetros: dataframe -> el dataframe de origen del que se quiere hacer el encoding
        columna -> la columna que se quiere transformar
        orden_valores -> el orden con el que se quiere hacer el mapeo (de más a menos importantes)
    output: el dataframe con las columnas encodeadas
    """
    modelo = OrdinalEncoder(categories = [orden_valores], dtype = int)
    transformados = modelo.fit_transform(df[[columna]])
    df[columna] = transformados
    with open(f'../data/modelo/encoding_{columna}.pkl', 'wb') as s:
        pickle.dump(modelo, s)
    
    return df

def metricas(y_test, y_train, y_test_pred, y_train_pred, tipo_modelo):
    """
    esta función evalua las variables respuesta iniciales y las predichas y calcula las métricas para evaluar los modelos de tipo lineal
    parámetros: y_test -> la variable respuesta original del dataframe de test (el que prueba)
        y_train -> la variable respuesta original del dataframe de train (el que entrena el modelo)
        y_test_pred -> la variable respuesta predicha del dataframe de test (el que prueba)
        y_train_pred -> la variable respuesta predicha del dataframe de test (el que prueba)
        tipo_modelo -> el nombre del modelo a evaluar (DecissionTree, RandomForest, GradientBoosting...)
    output: un dataframe con las métricas para el dataframe de test y el de train
    """
    resultados = {'MAE': [metrics.mean_absolute_error(y_test, y_test_pred), metrics.mean_absolute_error(y_train, y_train_pred)],
                'MSE': [metrics.mean_squared_error(y_test, y_test_pred), metrics.mean_squared_error(y_train, y_train_pred)],
                'RMSE': [np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)), np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))],
                'R2':  [metrics.r2_score(y_test, y_test_pred), metrics.r2_score(y_train, y_train_pred)],
                 "set": ["test", "train"]}
    df = pd.DataFrame(resultados)
    df["modelo"] = tipo_modelo
    return df

def linear_regression(X_train, y_train, X_test, y_test):
    """
    función para calcular la regresión lineal y almacenar el modelo
    input: X_train -> las variables predictoras del dataframe de entrenamiento
        y_train -> la variable respuesta del dataframe de entrenamiento
        X_test -> las variables predictoras del dataframe a testear
        y_test -> la variable respuesta del dataframe de testear
    output: y_pred_test -> el valor respuesta predicho del dataframe a entrenar
        y_pred_train -> el valor respuesta predicho del dataframe a testar
        results -> un dataframe con los resultados de la desviación entre el valor predicho y el valor real
    """
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)

    train_df = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train, 'Set': ['Train']*len(y_train)})
    test_df  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test,  'Set': ['Test']*len(y_test)})
    results = pd.concat([train_df,test_df], axis = 0)
    results['residual'] = results['Real'] - results['Predicted']
    with open(f'../data/modelo/modelo_lr.pkl', 'wb') as modelo:
        pickle.dump(modelo, modelo)
    return y_pred_test, y_pred_train, results

def decission_tree_params(X_train, y_train, X_test, y_test):
    """
    función para obtener las max_features (máximo número de elementos) y max_depth (la profundidad que tiene el árbol), que son los parámetros que luego se introducen para poder hacer las predicciones
    input: X_train -> las variables predictoras del dataframe de entrenamiento
        y_train -> la variable respuesta del dataframe de entrenamiento
        X_test -> las variables predictoras del dataframe a testear
        y_test -> la variable respuesta del dataframe de testear
    output: y_pred_test -> el valor respuesta predicho del dataframe a entrenar
        y_pred_train -> el valor respuesta predicho del dataframe a testar
        max_features -> máximo número de variables necesarias
        max_depth -> máxima profundidad del modelo
    """
    # create a regressor object
    modelo = DecisionTreeRegressor(random_state = 0) 
    
    # fit the regressor with X and Y data
    modelo.fit(X_train, y_train)

    max_features = np.sqrt(len(X_train.columns))
    max_depth = modelo.tree_.max_depth

    y_pred_test_dt = modelo.predict(X_test)
    y_pred_train_dt = modelo.predict(X_train)
    return y_pred_test_dt, y_pred_train_dt, max_features, max_depth

def modelos_grid_search(X_train, y_train, X_test, y_test, max_depth, max_features, input):
    """
    función para obtener los mejores parámetros (max depth, max_features, min_sample_split, min_samples_leaf) para hacer la predicción, según el modelo que se haya elegido
    input: X_train -> las variables predictoras del dataframe de entrenamiento
        y_train -> la variable respuesta del dataframe de entrenamiento
        X_test -> las variables predictoras del dataframe a testear
        y_test -> la variable respuesta del dataframe de testear
        max_depth -> profundidad máxima resultado del decision_tree_params
        max_features -> número máximo de las variables a tener en cuenta para el cálculo
        input -> modelo del que se quiere sacar los mejores parámetros:
            -DecissionTree
            -RandomForest
            -GradientBoosting
    output: best_estimator -> el objeto de scikitlearn con los mejores parámetros
        params -> los parámetros usados en el gridsearch
    """
    print(f"Start time: {datetime.now()}")

    # Define dictionary with key-value pairs of estimator class and params
    model_and_params = {
        'DecisionTree': (DecisionTreeRegressor(), {
            'max_depth' : list(range(1, int(max_depth)+1)),
            'max_features' : list(range(1, int(max_features)+1)),
            "min_samples_split": [10, 25, 50, 100, 150, 200], 
            "min_samples_leaf": [10, 25, 50, 100, 150, 200]
        }),

        'RandomForest': (RandomForestRegressor(), {
            'max_depth' : list(range(1, int(max_depth)+1)),
            'max_features' : list(range(1, int(max_features)+1)),
            "min_samples_split": [50, 100, 150, 200], 
            "min_samples_leaf": [50, 100, 150, 200]
        }),

        'GradientBoosting': (GradientBoostingRegressor(), {
            'max_depth' : list(range(1, int(max_depth)+1)),
            'max_features' : list(range(1, int(max_features)+1)),
            "min_samples_split": [50, 100, 150, 200], 
            "min_samples_leaf": [50, 100, 150, 200]
        })
    }

    if input not in model_and_params:
        print('Invalid input')

    # Unpack the tuple containing the estimator and params
    est, params = model_and_params[input]

    # Create GridSearchCV object using the estimator and parameters
    gs = GridSearchCV(
        estimator=est,
        param_grid=params,
        cv=10,
        verbose=-2,
        n_jobs=-1,
        return_train_score=True,
        scoring="neg_mean_squared_error"
    )

    # Fit the model on training data
    gs.fit(X_train, y_train)

    if input == 'DecisionTree':
        fig = plt.figure(figsize=(12, 6))
        tree.plot_tree(gs.best_estimator_, feature_names=X_train.columns, filled=True);

    print(f"End time: {datetime.now()}")
    return gs.best_estimator_, params

def modelo_prediccion(X_train, y_train, X_test, y_test, depth, features, samples_split, samples_leaf, input):
    """
    función para hacer el modelo de predicción de tipo DecissionTree, ReandomForest
    input: X_train -> las variables predictoras del dataframe de entrenamiento
        y_train -> la variable respuesta del dataframe de entrenamiento
        X_test -> las variables predictoras del dataframe a testear
        y_test -> la variable respuesta del dataframe de testear
        max_depth -> profundidad obtenida del mejor modelo
        max_features -> número máximo de las variables obtenidas del mejor modelo
        min_samples_split -> profundidad obtenida del mejor modelo
        min_samples_leaf -> número máximo de las variables obtenidas del mejor modelo
        input -> modelo del que se quiere sacar los mejores parámetros:
            -DecissionTree
            -RandomForest
            -GradientBoosting
    output: y_pred_test -> el valor respuesta predicho del dataframe a entrenar
        y_pred_train -> el valor respuesta predicho del dataframe a testar
    """
    if input == 'DecisionTree':
        modelo = DecisionTreeRegressor( max_depth =  depth, 
                                max_features=features, 
                                min_samples_split=samples_split, 
                                min_samples_leaf=samples_leaf)
    elif input == 'RandomForest':
        modelo = RandomForestRegressor( max_depth =  depth, 
                                max_features=features, 
                                min_samples_split=samples_split, 
                                min_samples_leaf=samples_leaf)
    elif input == 'GradientBoosting':
        modelo = GradientBoostingRegressor( max_depth =  depth, 
                                max_features=features, 
                                min_samples_split=samples_split, 
                                min_samples_leaf=samples_leaf)
    modelo.fit(X_train, y_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)

    #train_df = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train, 'Set': ['Train']*len(y_train)})
    #test_df  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test,  'Set': ['Test']*len(y_test)})
    #results = pd.concat([train_df,test_df], axis = 0)
    #results['residual'] = results['Real'] - results['Predicted']
    with open(f'../data/modelo/modelo_{input}.pkl', 'wb') as modelo_:
        pickle.dump(modelo, modelo_)
    return y_pred_test, y_pred_train

def knn_crossvalscore(X, y):
    """
    esta función calcula el número de vecinos óptimo para realizar el modelo KNeighbors.
    para ello toma la variable respuesta y las variables predictoras y devuelve un dataframe con los tres mejores registros
    input: X -> las variables predictoras
        y -> la variable respuesta
    output: dataframe -> con los resultados de los mejores vecinos ordenados por el NMSE
    """
    knn_scores =[]
    for k in range(1,21):
        # por defecto nos devuelve la precisión
        score = cross_val_score(KNeighborsRegressor(n_neighbors = k),
                            X = X,
                            y = y,
                            cv=10, 
                            scoring = "neg_mean_squared_error")
        knn_scores.append(score.mean())
    knn = pd.DataFrame(knn_scores, range(1,21)).reset_index()
    knn.columns = ["number_neighbors", "score"]
    #knn.sort_values(by = "score", ascending = False).head(3)
    return knn.sort_values(by = "score", ascending = False).head(3)

def modelo_knn(X_train, y_train, X_test, y_test, neighbors):
    """
    esta función sirve para hacer la prodección con el modelo KNeighbors.
    input: X_train -> las variables predictoras del dataframe de entrenamiento
        y_train -> la variable respuesta del dataframe de entrenamiento
        X_test -> las variables predictoras del dataframe a testear
        y_test -> la variable respuesta del dataframe de testear
        negihbors -> el número de vecinos necesarios para hacer la predicción
    output: y_pred_test -> el valor respuesta predicho del dataframe a entrenar
        y_pred_train -> el valor respuesta predicho del dataframe a testar
    """
    knn = KNeighborsRegressor(n_neighbors = neighbors)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    with open(f'../data/modelo/modelo_knn.pkl', 'wb') as modelo:
        pickle.dump(knn, modelo)
    return y_pred_test, y_pred_train

def transformers_input(encoding_clarity, encoding_color, encoding_cut, estandarizacion, modelo):
    """
    función que importa los modelos necesarios para hacer la predicción. para ello, se importa el encoding, estandarización y el modelo.
    input: encoding_clarity -> el nombre del archivo que aloja el encoding
        encoding_color -> el nombre del archivo que aloja el encoding
        encoding_cut -> el nombre del archivo que aloja el encoding
        estandarizacion -> nombre del archivo que aloja el objeto de estandarización
        modelo -> el objeto que realizó la predicción, y que está entrenado con los resultados
    """
    # encoding clarity
    with open(f'../data/modelo/{encoding_clarity}.pkl', 'rb') as clarity:
        encoding_clarity = pickle.load(clarity)
        
    # encoding color
    with open(f'../data/modelo/{encoding_color}.pkl', 'rb') as color:
        encoding_color = pickle.load(color)
        
    # encoding cut
    with open(f'../data/modelo/{encoding_cut}.pkl', 'rb') as cut:
        encoding_cut = pickle.load(cut)

    # estandarización
    with open(f'../data/modelo/{estandarizacion}.pkl', 'rb') as estandarizacion:
        estandarizacion = pickle.load(estandarizacion)
    
    # modelo
    with open(f'../data/modelo/{modelo}.pkl', 'rb') as modelo:
        modelo = pickle.load(modelo)
    return encoding_clarity, encoding_color, encoding_cut, estandarizacion, modelo