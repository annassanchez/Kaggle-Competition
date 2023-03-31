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

This is the third project of the Ironhack's Data Analysis Bootcamp.
<br>The dataset is available [here](https://www.kaggle.com/competitions/diamonds-datamad1022/data). Also [here](https://www.kaggle.com/competitions/diamonds-datamad1022) is the link to the Kaggle competition.
<br>The aim of this project is to develop a machine learning model pipeline to predict the price of diamonds given the dataset and submit it to the competition. So basically, is to win the competition with the most accurate prediction!

# Data

## Diamond's 4Cs
The price of diamonds is given by the 4C's:
- `carat`: the diamond weight. 
<br>One carat weighs 1/5 of a gram and is divided into 100 points, so a diamond weighing 1.07 ct. is referred to as "one carat and seven points."
<br>Diamonds of the same weight doesn't mean that the size appearance i the same is the same -> other factors as the `cut` or the material might affect also.
- `cut`: the quality of the diamond's cut. The cut is measures the diamond's refeaction - evaluating how the light angle shifts through the diamond. The better the cut, the diamond is clearer and shines brighter, therfore is more expensive. If the cut is not well enough, the diamond will be more matt and
- `color`: the color of the diamond. The color is coded from `D` to `Z`, according to how clear it is. 
<br>To determine the correct color, all submitted diamonds are compared to an internationally accepted master set of stones, the colors of which range from `D`, or colorless (the most sought after) to `Z`, the most yellow/brown - aside from "fancy" yellow or brown.
- `clarity`: a measurement of how clear the diamond is.
<br>There are two types of clarity characteristics: inclusions and blemishes. In order to grade the clarity of a diamond, it is necessary to observe the number and nature of external and internal characteristics of the stone, as well as their size and position. The difference is based on their locations: inclusions are enclosed within a diamond, while blemishes are external characteristics. IGI grading reports show plotted diagrams of clarity characteristics marked in red for internal and green for external features.

![](https://www.igi.org/assets/images/diamond-4cs.jpg)
Each of the C's is graded on a scale and can be evaluated for quality. The final price is given by an equilibrium of all of the variables - the higher they rank in all the variables, the better.


## variables
The file contains the following variables:
- `id`: only for test & sample submission files, id for prediction sample identification
- `price`: price in USD. Is the variable that needs to be predicted in the `test.csv` file.
- `carat`: weight of the diamond, measured in qt.
- `cut`: quality of the cut, from worst to best (Fair, Good, Very Good, Premium, Ideal).
- `color`: diamond colour, from whiter to yellower (D, E, F, G, H, I,J)
- `clarity`: a measurement of how clear the diamond is, according to the internal and superficial damages ('I3', 'I2', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF' , 'FL').
- `x`: length in mm.
- `y`: width in mm.
- `z`: depth in mm.
- `depth`: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79).
- `table`: width of top of diamond relative to widest point (43--95).

# Challenge

So the challenge is to predict the `price` of the diamonds given the variables presented above.
<br>Here's a dashboard that protrays the results.

# Toolkit

- [**pandas**](https://pypi.org/project/pandas/): this library is used to work with related and table like data structures.
- [**numpy**](https://pypi.org/project/numpy/): library used for scientific calculation.
- [**pickle**](https://docs.python.org/3/library/pickle.html): a module that generates files that can be used within python to store any kind of data -- from dataframes to dicionaries and so on.
- [**math**](https://docs.python.org/3/library/math.html): this module provides access to the mathematical functions - such as ceil.
- [**scipy**](https://scipy.org/): this module provides fudamental algorithms for scientific computing in Python -- algorithms for optimization, integration, optimization, statistics..., those are the ones being used.
- [**datetime**](https://docs.python.org/3/library/datetime.html): this module supplies classes for manipulating dates and times - used mostly for the date tracking.
- [**IPython**](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html): IPython means interactive Python. It is an interactive command-line terminal for Python.
- [**matplotlib.pyplot**](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html): is an interface of matplotlib. It provides an implicit way of plotting -- similar to MATLAB.
- [**seaborn**](https://seaborn.pydata.org/): is a python data visualization library based on matplotlib. It provides a high-level interface for  drawing attractive nad informative statistical graphics.
- [**scikit-learn**](https://scikit-learn.org/stable/): built on numpy, scipy andd matplotlib, this library provides simple and efficient tools for predictive data analysis.
- [**warnings**](https://docs.python.org/3/library/warnings.html): this library helps to hide the annoying warnings that python sometimes throws.

# Status

Version 1.0 > There are some inifinite prints that need to be refined on some of the models. 

# Contact

Feel free to contact me [here](mailto:annassanchez@gmail.com) if you want to know more or find any mistakes.