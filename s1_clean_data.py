# encoding=utf-8

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

clean_data_file = "../data/house_price/tmp/clean_data.csv"

data_train = pd.read_csv('../data/house_price/input/train.csv')
data_test = pd.read_csv('../data/house_price/input/test.csv')

data_train['is_train'] = True
data_test['is_train'] = False

data = pd.concat([data_train, data_test])
#print data.info()


# 缺失值处理

def show_missing():
    missing = data.columns[data.isnull().any()].tolist()
    print data[missing].isnull().sum()

    # 数值型特征缺失值可能为0
    print "---categorial------------------------------"
    categorial_features = data.select_dtypes(include=[object])
    numeric_features = data.select_dtypes(exclude=[object])

    missing = categorial_features.columns[categorial_features.isnull().any()].tolist()
    print categorial_features[missing].isnull().sum()

    print "---numeric------------------------------"
    # 数值型特征缺失值可能为0,

print "缺失值："
show_missing()

# Looking at categorical values, 统计输出某个属性每种值的个数
def cat_exploration(column):
    return data[column].value_counts()

# Imputing the missing values
def cat_imputation(column, value):
    data.loc[data[column].isnull(),column] = value

def LotFrontage_LotArea():
    '''
    1、缺失值。查看LotFrontage与sqrt(LotArea)相关性，用相关性高的值代替LotFrontage的缺失值
    :return:
    '''
    #data['LotFrontage'].corr(data['LotArea'])
    data['SqrtLotArea'] = np.sqrt(data['LotArea'])
    #data['LotFrontage'].corr(data['SqrtLotArea'])
    cond = data['LotFrontage'].isnull()
    data.LotFrontage[cond] = data.SqrtLotArea[cond]
    del data['SqrtLotArea']

def Alley():
    #cat_exploration('Alley')
    cat_imputation('Alley','None')

def MasVnr():
    #data[['MasVnrType', 'MasVnrArea']][data['MasVnrType'].isnull() == True]
    #cat_exploration('MasVnrType')
    cat_imputation('MasVnrType', 'None')
    cat_imputation('MasVnrArea', 0.0)

def Basement():
    basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
    #houseprice[basement_cols][houseprice['BsmtQual'].isnull()==True]
    for cols in basement_cols:
        if 'FinSF' not in cols:
            cat_imputation(cols, 'None')
def Electrical():
    # cat_exploration('Electrical')
    cat_imputation('Electrical', 'SBrkr')

def Fireplace():
    # cat_exploration('FireplaceQu')
    # data['Fireplaces'][data['FireplaceQu'].isnull() == True].describe()
    cat_imputation('FireplaceQu', 'None')
    # pd.crosstab(data.Fireplaces, data.FireplaceQu)

def Garages():
    garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
    #data[garage_cols][data['GarageType'].isnull() == True]
    for cols in garage_cols:
        if data[cols].dtype == np.object:
            cat_imputation(cols, 'None')
        else:
            cat_imputation(cols, 0)

def Pool():
    # cat_exploration('PoolQC')
    # data['PoolArea'][data['PoolQC'].isnull() == True].describe()
    cat_imputation('PoolQC', 'None')

def Fence():
    cat_imputation('Fence', 'None')

def MiscFeature():
    cat_imputation('MiscFeature', 'None')


def Street_Utilities():
    #print(data['Street'].value_counts())
    #print(data['Utilities'].value_counts())
    to_remove = ['Street', 'Utilities']
    data.drop(to_remove, axis=1, inplace=True)



LotFrontage_LotArea()
Alley()
MasVnr()
Basement()
Electrical()
Fireplace()
Garages()
Pool()
Fence()
MiscFeature()

Street_Utilities()

print "所有属性：\n", data.columns

data.to_csv(clean_data_file, index=False)



