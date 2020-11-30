import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import os
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def Estimation(M, num_level):
    '''
    Estimate the parameters for the demand model.
    '''
    raw_data = pd.read_excel(r'data/michigan-history.xlsx').fillna(axis=1,method='ffill')

    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data.set_index("date", inplace=True)

    data = raw_data.resample('W').sum()
    
    indent = M/num_level
    tv = data['totalTestsViral'].to_numpy()
    X_train = tv * (M-tv)
    X_train = np.transpose(np.expand_dims(X_train[1:], 0))

    y_train = (tv[1:]//indent)*indent-(tv[:-1]//indent)*indent

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    sigma = np.linalg.norm(y_pred-y_train)/np.sqrt(np.array([len(y_pred)+1e-10]))
    resdict = {"a": model.coef_[0], "b": 1, "c": model.intercept_, "d": sigma[0]}

    dirs = 'middle-results/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    doc = open("middle-results/inference-results.txt", "w")
    print(resdict, file=doc)
    doc.close()

    return resdict
