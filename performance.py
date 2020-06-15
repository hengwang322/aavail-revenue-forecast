#!/usr/bin/env python
"""
a collection of functions to measure & visualize performance
"""

import os, pickle, re
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import plotly.graph_objects as go
with open('__version__','r+') as f:
    MODEL_VERSION = f.read()
    f.close

# from model import get_preprocessor

def percentage_error(actual, predicted):
    '''
    given 2 arrays and remove entries in the array if 0
    '''
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape(y_true, y_pred):
    '''
    caculate mean absolute percentage error, ignoring 0 entry.
    '''
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

def plot_ts(country):
    '''
    plot out the y_true vs y_pred, given the country, all_data, and all_models
    '''
    version_ = re.sub("\.","_",str(MODEL_VERSION))
    all_data, all_models = pickle.load(open(os.path.join("models",f"all_data_model-{version_}.pickle"), "rb" ))
    y_true = all_data[country]['y']
    y_pred = all_models[country].predict(all_data[country]['X'])
    all_dates = all_data[country]['dates']
    rmse_ = round(mse(y_true,y_pred,squared=False),2)
    mae_ = round(mae(y_true,y_pred),2)
    mape_ = round(mape(y_true,y_pred),2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_dates, y=y_true, name='Actual Revenue'))
    fig.add_trace(go.Scatter(x=all_dates, y=y_pred, name='Predicted Revenue'))

    fig.update_layout(title=f"{country.replace('_',' ').title()}: RMSE:{rmse_}, MAE:{mae_}, MAPE:{mape_}%",
                      yaxis_title="Revenue")

    fig.show()

def show_importance(country):
    '''
    returns a df that shows feature importance
    '''
    version_ = re.sub("\.","_",str(MODEL_VERSION))
    all_data, all_models = pickle.load(open(os.path.join("models",f"all_data_model-{version_}.pickle"), "rb" ))
    df = pd.DataFrame.from_dict({'feature':all_data[country]['X'].columns,
                        'importance':all_models[country].best_estimator_.steps[1][1].feature_importances_})\
                        .sort_values(by='importance',ascending=False)\
                        .style\
                        .bar(color='lightblue', subset=['importance'], align='zero')
    return df

def compare_drift(X_src,y_src,X_new,y_new):
    clf_y = EllipticEnvelope(random_state=0,contamination=0.01)
    clf_X = EllipticEnvelope(random_state=0,contamination=0.01)

    clf_X.fit(X_src)
    clf_y.fit(y_src.reshape(y_src.size,1))

    test_X = clf_X.predict(X_new)

    test_y = clf_y.predict(y_new.reshape(-1, 1))

    X_distance = wasserstein_distance(X_src.values.flatten(),X_new.values.flatten())

    y_distance = wasserstein_distance(y_src.flatten(),y_new.flatten())

    X_outlier = len(test_X[test_X == -1])/len(test_X)

    y_outlier = len(test_y[test_y == -1])/len(test_y)

    results = {
        'X_wasserstein_distance':X_distance,
        'y_wasserstein_distance':y_distance,
        'X_outlier_percentage':X_outlier,
        'y_outlier_percentage':y_outlier
    }

    return results
