#encoding=utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing

clean_data_file = "../data/house_price/tmp/clean_data.csv"
feature_data_file = "../data/house_price/tmp/feature_data.csv"

data = pd.read_csv(clean_data_file)

# 添加新特征


def time_relative():
    # ['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold']

    Built_Sold_Gap = data['YrSold'] - data['YearBuilt']
    data['Built_Sold_Gap'] = Built_Sold_Gap

    Remod_Sold_Gap = data['YrSold'] - data['YearRemodAdd']
    data['Remod_Sold_Gap'] = Remod_Sold_Gap

    Built_Remod_Gap = data['YearRemodAdd'] - data['YearBuilt']
    data['Built_Remod_Gap'] = Built_Remod_Gap

    # 计算售卖年份与建造年份时间间隔的平均值，将间隔与平均值差作为特征
    mean_built_sold_gap = Built_Sold_Gap - Built_Sold_Gap.mean()
    data['mean_built_sold_gap'] = mean_built_sold_gap

    mean_remod_sold_gap = Remod_Sold_Gap - Remod_Sold_Gap.mean()
    data['mean_remod_sold_gap'] = mean_remod_sold_gap

    mean_built_remod_gap = Built_Remod_Gap - Built_Remod_Gap.mean()
    data['mean_built_remod_gap'] = mean_built_remod_gap

    # 月份用季节表示，变为类别型
    mosold = data['MoSold']
    season = []
    for m in mosold:
        if m == 0:
            season.append("none")
        elif m <= 3:
            season.append("spring")
        elif m <= 6:
            season.append("summer")
        elif m <= 9:
            season.append("autumn")
        else:
            season.append("winter")
    data['SeaSold'] = season

    data['MoSold'] = data['MoSold'].astype(str)
    data['YearBuilt'] = data['YearBuilt'].astype(str)
    data['YearRemodAdd'] = data['YearRemodAdd'].astype(str)
    data['YrSold'] = data['YrSold'].astype(str)

    # 组合售卖的年份和月份作为特征
    ysold_msold = data['YrSold'] + "-"+ data['MoSold']
    data['ysold_msold'] = ysold_msold

# def area_relative():
    
time_relative()

base_columns = ['Id', 'is_train', 'SalePrice']
features = data[base_columns]

# 类别特征，onehot编码
# print data.select_dtypes(include=[object]).columns.values

# a = ["'" + x + "'" for x in data.select_dtypes(include=[object]).columns.values]
# print ", ".join(a)

# categorial_features = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
#                        'BsmtFinSF1','BsmtFinSF2', 'Alley', 'MasVnrType',
#                        'Electrical', 'FireplaceQu', 'GarageType', 'GarageQual'
#                        ]

# 删除的属性PoolQC、MiscFeature
categorial_features = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                       'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2',
                       'Electrical', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
                       'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond',
                       'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC',
                       'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig',
                       'LotShape', 'MSZoning', 'MasVnrType', 'Neighborhood',
                       'PavedDrive', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType',

                       'SeaSold', 'MoSold','YearBuilt', 'YearRemodAdd', 'YrSold', 'ysold_msold',

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

# a = ["'" + x + "'" for x in  data.select_dtypes(exclude=[object]).columns.values]
# print ", ".join(a)
# numeric_features = ['LotFrontage', 'MasVnrArea', 'LotArea', 'GarageYrBlt', 'GarageCars',
#                     'GarageArea'
#                     ]

numeric_features = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces',
                    'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath',
                    'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MSSubClass',
                    'MasVnrArea', 'MiscVal', 'OpenPorchSF', 'OverallCond', 'OverallQual',
                    'PoolArea', 'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF',

                    'Built_Sold_Gap', 'Remod_Sold_Gap', 'Built_Remod_Gap', 'mean_built_sold_gap',
                    'mean_remod_sold_gap', 'mean_built_remod_gap'
                    ]

# print data[numeric_features].head()
numeric_data = data[numeric_features]
print "剩余的缺失值："
cnt = 0
for column in numeric_features:
    if len (data[data[column].isnull()]) > 0:
        cnt += 1
        print column
    numeric_data[column].fillna(data[column].value_counts().idxmax(), inplace=True)
print "剩余的缺失值cnt：", cnt

scaler = preprocessing.StandardScaler()
numeric_data.loc[:, numeric_features] = scaler.fit_transform(numeric_data)
features = pd.concat([features, numeric_data], axis=1)
print features.head()
print 'after numeric:', features.shape


features.to_csv(feature_data_file, index=False)