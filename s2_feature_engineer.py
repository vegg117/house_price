#encoding=utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing

clean_data_file = "../data/house_price/tmp/clean_data.csv"
feature_data_file = "../data/house_price/tmp/feature_data.csv"

data = pd.read_csv(clean_data_file)

# 添加新特征


# 去除离群点



base_columns = ['Id', 'is_train', 'SalePrice']
features = data[base_columns]

# 类别特征，onehot编码
print data.select_dtypes(include=[object]).columns.values
categorial_features = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                       'BsmtFinSF1','BsmtFinSF2', 'Alley', 'MasVnrType',
                       'Electrical', 'FireplaceQu', 'GarageType', 'GarageQual'
                       ]
print 'before categorial:', features.shape
categorial_data = pd.get_dummies(data[categorial_features])
# print categorial_data.head()
# print data[categorial_features].head()
features = pd.concat([features, categorial_data], axis=1)
# data.drop(categorial_features, axis=1, inplace=True)
# print data[categorial_features].head()
print 'after categorial:', features.shape

# 数值特征， 归一化
print 'before numeric:', features.shape
print data.select_dtypes(exclude=[object]).columns.values
numeric_features = ['LotFrontage', 'MasVnrArea', 'LotArea', 'GarageYrBlt', 'GarageCars',
                    'GarageArea'
                    ]
numeric_data = data[numeric_features]
scaler = preprocessing.StandardScaler()
numeric_data.loc[:, numeric_features] = scaler.fit_transform(numeric_data)
features = pd.concat([features, numeric_data], axis=1)
print features.head()
print 'after numeric:', features.shape


features.to_csv(feature_data_file, index=False)