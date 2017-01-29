# encoding=utf-8

import datetime
import time
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import linear_model, model_selection
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import roc_auc_score


feature_data_file = "../data/house_price/tmp/feature_data.csv"

data = pd.read_csv(feature_data_file)

X = data[data['is_train']].drop(['Id', 'SalePrice'], axis=1)
y = data[data['is_train']]['SalePrice']
print X.shape
print y.shape

X_test = data[~data['is_train']].drop(['Id', 'SalePrice'], axis=1)
Id_test = data[~data['is_train']]['Id']
print X_test.shape
print Id_test.shape


print X


def rmsle_score(y, p):
    return -np.sqrt(np.sum((np.log(1+y) - np.log(1+p))**2)/y.shape[0])
rmsle = metrics.make_scorer(rmsle_score)

# Ridge regression: Count RMSLE on cross-validation
param_grid = {
              'alpha': [0.5, 1, 2, 6, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150],
             }

ridge = linear_model.Ridge()
ridge_gs = model_selection.GridSearchCV(ridge, param_grid, cv=3, scoring=rmsle)
ridge_gs.fit(X, y)
print(ridge_gs.best_score_)
print(ridge_gs.best_params_)

