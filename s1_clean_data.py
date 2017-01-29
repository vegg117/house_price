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

def missing_value():
    missing = data.columns[data.isnull().any()].tolist()

    # print "---categorial------------------------------"
    # categorial_features = data.select_dtypes(include=[object])
    # numeric_features = data.select_dtypes(exclude=[object])
    #
    # missing = categorial_features.columns[categorial_features.isnull().any()].tolist()
    # print categorial_features[missing].isnull().sum()
    #
    # print "---numeric------------------------------"
    # # 数值型特征缺失值可能为0,
    # missing = numeric_features.columns[numeric_features.isnull().any()].tolist()
    # print numeric_features[missing].isnull().sum()
    return missing


print "缺失值："
print data[missing_value()].isnull().sum()
# exit()

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


def Fence():
    cat_imputation('Fence', 'None')


# MiscFeature' and 'PoolQC' have more than 96% nan values, so we can remove them
def MiscFeature():
    data.drop(['MiscFeature'], axis=1, inplace=True)

def PoolQC():
    # cat_exploration('PoolQC')
    # data['PoolArea'][data['PoolQC'].isnull() == True].describe()
    data.drop(['PoolQC'], axis=1, inplace=True)


def Street_Utilities():
    #print(data['Street'].value_counts())
    #print(data['Utilities'].value_counts())
    to_remove = ['Street', 'Utilities']
    data.drop(to_remove, axis=1, inplace=True)


# 剩余缺失值的处理
def remainLossValue():
    print "remain loss value feature:"
    columns = data.columns.values
    cnt = 0
    for col in columns:
        if(len(data[data[col].isnull()]) > 0):
            cnt += 1
            print col
        if data[col].dtype == np.object:
            cat_imputation(col, 'None')
        else:
            cat_imputation(col, 0)
    print "the mumber of reamin loss value feature is ", cnt


# 按行统计每个样本的属性缺失值的个数，剔除离群点
def line_missing():
    loss_count = pd.DataFrame(columns=['Id', 'loss'])
    ids = data['Id']
    print "总用户数：", len(ids)
    for id in ids:
        loss_count.loc[id, 'loss'] = len(data[data[data['Id'] == id].isnull()])
        exit()


print data.head()
exit()


# 缺失值处理
LotFrontage_LotArea()
Alley()
MasVnr()
Basement()
Electrical()
Fireplace()
Garages()
PoolQC()
Fence()
MiscFeature()
# remainLossValue()
Street_Utilities()

line_missing()
exit()

data.to_csv(clean_data_file, index=False)



