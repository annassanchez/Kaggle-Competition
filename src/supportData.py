import pickle
import pandas as pd
import numpy as np
import scipy

def importDatasets():
    df = pd.read_csv('./data/train.csv')
    return df