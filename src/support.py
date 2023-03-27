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
            with open(f'../data/modelo/encoding_{columna}.pkl', 'wb') as s:
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

def linear_regression(X_train, y_train, X_test, y_test):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)

    train_df = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train, 'Set': ['Train']*len(y_train)})
    test_df  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test,  'Set': ['Test']*len(y_test)})
    results = pd.concat([train_df,test_df], axis = 0)
    results['residual'] = results['Real'] - results['Predicted']
    return y_pred_test, y_pred_train, results

def decission_tree_params(X_train, y_train, X_test, y_test):
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
    print(datetime.now())
    param = {
        'max_depth' : list(range(1, int(max_depth)+1)),
        'max_features' : list(range(1, int(max_features)+1)),
        "min_samples_split": [10, 25, 50, 100, 150, 200], # [100, 150, 200] -> número de datos
        "min_samples_leaf": [10, 25, 50, 100, 150, 200]
    }
    if input == 'DecisionTree':
        gs = GridSearchCV(
                estimator=DecisionTreeRegressor(),
                param_grid= param,
                cv=10,
                verbose=-2, # muestra el progreso - 2 pa que no te saque todo el input
                n_jobs = -1,
                return_train_score = True,
                scoring="neg_mean_squared_error")
        gs.fit(X_train, y_train)
        fig = plt.figure(figsize=(12, 6))
        tree.plot_tree(gs.best_estimator_, feature_names=X_train.columns, filled=True);
    elif input == 'RandomForest':
        gs = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid= param,
            cv=10,
            verbose=-2, # muestra el progreso - 2 pa que no te saque todo el input
            n_jobs = -1,
            return_train_score = True,
            scoring="neg_mean_squared_error")
        gs.fit(X_train, y_train)
    elif input == 'GradientBoosting':
        gs = GridSearchCV(
            estimator=GradientBoostingRegressor(),
            param_grid= param,
            cv=10,
            verbose=-2, # muestra el progreso - 2 pa que no te saque todo el input
            n_jobs = -1,
            return_train_score = True,
            scoring="neg_mean_squared_error")
        gs.fit(X_train, y_train)
    else: 
        print('aprende a escribir')
    print(datetime.now())
    return gs.best_estimator_, param

def modelo_prediccion(X_train, y_train, X_test, y_test, max_depth, max_features, min_samples_split, min_samples_leaf, input):
    if input == 'DecisionTree':
        modelo = DecisionTreeRegressor( max_depth =  max_depth, 
                                max_features=max_features, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf)
    elif input == 'RandomForest':
        modelo = RandomForestRegressor(min_samples_split= min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           max_features=max_features,
                           max_depth=max_depth)
    elif input == 'GradientBoosting':
        modelo = GradientBoostingRegressor(min_samples_split= min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    max_depth=max_depth)
    modelo.fit(X_train, y_train)
    y_pred_test = modelo.predict(X_test)
    y_pred_train = modelo.predict(X_train)

    #train_df = pd.DataFrame({'Real': y_train, 'Predicted': y_pred_train, 'Set': ['Train']*len(y_train)})
    #test_df  = pd.DataFrame({'Real': y_test,  'Predicted': y_pred_test,  'Set': ['Test']*len(y_test)})
    #results = pd.concat([train_df,test_df], axis = 0)
    #results['residual'] = results['Real'] - results['Predicted']

    return y_pred_test, y_pred_train

def knn_crossvalscore(X, y):
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
    knn.sort_values(by = "score", ascending = False).head(3)
    return knn

def modelo_knn(X_train, y_train, X_test, y_test, neighbors):
    knn = KNeighborsRegressor(n_neighbors = neighbors)
    knn.fit(X_train, y_train)
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    return y_pred_test, y_pred_train