#import quandl
import pandas as pd
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
import os
from arch import arch_model
#from sklearn.decomposition import FactorAnalysis
#from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import copy

if __name__ == "__main__":
    raw_data = pd.read_csv(r'data/michigan-history.csv',
                                  encoding='gb2312').fillna(axis=1,method='ffill')

    raw_data['date'] = pd.to_datetime(raw_data['date'])
    raw_data.set_index("date", inplace=True)

    data = raw_data.resample('W').sum()
    #print(data)
    M = 5000000
    num_level = 20
    indent = M/num_level
    tv = data['totalTestResultsIncrease'].to_numpy()
    X_train = tv * (M-tv)
    #X_train = (X_train // indent) * indent
    X_train = np.transpose(np.expand_dims(X_train[1:], 0))

    #y_train = (tv[1:]//indent)*indent-(tv[:-1]//indent)*indent
    y_train = tv[1:]-tv[:-1]
    #y_train = (y_train // indent)*indent
    #y_train = y_train[1:]#.to_numpy()
    #print("x train")
    #print(X_train)
    #print("y train")


    #print("pct")
    #print(data['totalTestsViral'].pct_change())
    #print(data['totalTestsViral'][:])
    model = LinearRegression()
    model.fit(X_train, y_train)

    #print(model.intercept_)
    #print(model.coef_)
    y_pred = model.predict(X_train)
    y_pred_modified = copy.deepcopy(y_pred)
    y_pred_modified[0] = y_train[0]
    fig, axs = plt.subplots()
    xs = list(pd.date_range('2020-05-02', '2020-11-29', freq="W"))
    #print(len(xs))
    #print(len(y_train))
    axs.plot(xs, np.cumsum(y_train), label="weekly total test in Michigan")
    axs.plot(xs, np.cumsum(y_pred_modified), label="prediction")
    axs.set_xlabel('date')
    axs.set_ylabel('total test')
    axs.legend()
    dirs = 'figs/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    plt.savefig(dirs+"test-prediction.png",bbox_inches='tight')
    #plt.xticks(pd.date_range('2020-04-25', '2020-11-29'), rotation=90)
    #plt.show()
    sigma = np.linalg.norm(y_pred-y_train)/np.sqrt(np.array([len(y_pred)+1e-10]))
    resdict = {"a": model.coef_[0], "b": 1, "c": model.intercept_, "d": sigma[0]}

    dirs = 'middle-results/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    doc = open("middle-results/inference-results.txt","w")
    print(resdict, file=doc)
    doc.close()
    #print(sigma)
