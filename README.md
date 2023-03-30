# Kaggle Competition: Diamond Price Prediction

![yo, mirando los diamantes que me podría comprar si ganara](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg/640px-Audrey_Hepburn_in_Breakfast_at_Tiffany%27s.jpg)

# Index
1. `data`
    - `data/modelo` -> output from the different tranformer objects and machine learning model
        - `encoding_clarity` -> encoding object for the `clarity` column
        - `encoding_cut` -> encoding object for the `cut` column
        - `encoding_color` -> encoding object for the `color` column
        - `estandarizacion` -> standarization encoder 
        - `modelo_DecissionTree` -> model object for the DecissionTree prediction
        - `modelo_GradientBoosting` -> model object for the GradientBoosting prediction
        - `modelo_knn` -> model object for the KNN prediction
        - `modelo_RandomForest` -> model object for the RandomForest prediction
    - `data/submission` -> the folder that stores all the submissions
    - [`train.csv`](data/train.csv) -> input file for training the model
    - [`test.csv`](data/test.csv) -> input file that holds the data that needs to be predicted
    - [`sample_submission.csv`](data/sample_submission.csv) -> an example of how the submission should look like.
2. `images`
3. `notebooks`
    - [`1_EDA`](notebooks/1_EDA.ipynb) -> notebook that contains the exploratory data analysis
    - [`2_preprocessing_predictions`](notebooks/2_preprorcessing_predictions.ipynb) -> notebook that contains the actual machine learning models and transformations
    - [`3_prediction`](notebooks/3_prediction.ipynb) -> notebooks that transforms the test.csv in order to make the price prediction and prepares the submission output.
4. `src`
    - [`support.py`](src/support.py) -> support file that contains all the functions for the charts, the transformations and models.

# Context
The dataset is available [here](https://www.kaggle.com/competitions/diamonds-datamad1022/data). Also [here](https://www.kaggle.com/competitions/diamonds-datamad1022) is the link to the Kaggle competition.
<br>The aim of this project is to develop a machine learning model pipeline to predict the price of diamonds given the dataset and submit it to the competition. So basically, is to win the competition with the most accurate prediction!

# Data
## meta

## variables
