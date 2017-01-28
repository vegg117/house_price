# encoding=utf-8

import pandas as pd
import numpy as np

clean_data_file = "../data/house_price/tmp/clean_data.csv"

data_train = pd.read_csv('../data/house_price/input/train.csv')
data_test = pd.read_csv('../data/house_price/input/test.csv')

data_train['is_train'] = True
data_test['is_train'] = False

data = pd.concat([data_train, data_test])
#print data.info()

#print len(data[data['is_train'] == False]) + len(data[data['is_train'] == True])

# 缺失值处理

# Looking at categorical values
def cat_exploration(column):
    return data[column].value_counts()

# Imputing the missing values
def cat_imputation(column, value):
    data.loc[data[column].isnull(),column] = value

def LotFrontage_LotArea():
    cond = data['LotFrontage'].isnull()
    data.LotFrontage[cond] = data.SqrtLotArea[cond]
    del data['SqrtLotArea']

def Fence():
    cat_imputation('Fence', 'None')

Fence()

data.to_csv(clean_data_file, index=False)



