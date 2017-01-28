#encoding=utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing

clean_data_file = "../data/house_price/tmp/clean_data.csv"
feature_data_file = "../data/house_price/tmp/feature_data.csv"

data = pd.read_csv(clean_data_file)

# 添加新特征



# 类别特征，onehot编码
categorial_features = ['BsmtExposure']

print 'before categorial:', data.shape
categorial_data = pd.get_dummies(data[categorial_features])
# print categorial_data.head()
# print data[categorial_features].head()
data = pd.concat([data, categorial_data], axis=1)
data.drop(categorial_features, axis=1, inplace=True)
# print data[categorial_features].head()
print 'after categorial:', data.shape

# 数值特征， 归一化
numeric_features = ['1stFlrSF', '2ndFlrSF']
scaler = preprocessing.StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])
print data.head()



data.to_csv(feature_data_file, index=False)