import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Функции
def value_to_percent(df):
    array = (df.isnull().sum() / len(df)) * 100
    array = array[array > 0].sort_values()
    return array


# ОБУЧАЮЩИЙ НАБОР ДАННЫХ

df_train = pd.read_csv('train.csv')
fall_index_train = df_train.columns[df_train.isnull().any()].tolist()

# print(df_train.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False))
# print()
# print()
# print()
# Очистка выбросов
array_drop = df_train[(df_train['OverallQual'] > 8) & (df_train['SalePrice'] < 200000)].index
df_train = df_train.drop(array_drop, axis=0)
df_train = df_train.drop(
    df_train[(df_train['OverallQual'] == 4) & (df_train['SalePrice'].between(200000, 300000))].index, axis=0)
df_train = df_train.drop(df_train[df_train['SalePrice'] > 700000].index, axis=0)
df_train = df_train.drop([581, 825, 1061, 1190, 809], axis=0)
df_train = df_train.drop(529, axis=0)
df_train = df_train.drop(53, axis=0)
df_train = df_train.drop(635, axis=0)
df_train = df_train.drop(
    [897, 910, 1292, 1416, 185, 1169, 624, 773, 1230, 1300, 1334, 1379, 234, 650, 936, 973, 977, 1243, 1278], axis=0)

# Работа с отсутствующими данными
df_train = df_train.drop('Id', axis=1)
df_train = df_train.drop('PoolQC', axis=1)
df_train = df_train.drop('MiscFeature', axis=1)
df_train = df_train.drop('Alley', axis=1)
df_train = df_train.drop('Fence', axis=1)
df_train['MasVnrType'] = df_train['MasVnrType'].fillna('None')
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('None')
df_train['GarageCond'] = df_train['GarageCond'].fillna('None')
df_train['GarageQual'] = df_train['GarageQual'].fillna('None')
df_train['GarageFinish'] = df_train['GarageFinish'].fillna('None')
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna('None')
df_train['GarageType'] = df_train['GarageType'].fillna('None')
df_train['BsmtQual'] = df_train['BsmtQual'].fillna('None')
df_train['BsmtCond'] = df_train['BsmtCond'].fillna('None')
df_train['BsmtFinType1'] = df_train['BsmtFinType1'].fillna('None')
df_train['BsmtExposure'] = df_train['BsmtExposure'].fillna('None')
df_train['BsmtFinType2'] = df_train['BsmtFinType2'].fillna('None')
df_train['LotFrontage'] = df_train.groupby('Neighborhood')['LotFrontage'].transform(
    lambda value: value.fillna(value.mean()))

# Масштабирование
df_train['MiscVal'] = (df_train['MiscVal'] - df_train['MiscVal'].min()) / (
            df_train['MiscVal'].max() - df_train['MiscVal'].min())

# Работа с категориальными атрибутами
df_train['MSSubClass'] = df_train['MSSubClass'].apply(str)
categ = df_train.select_dtypes(include='object')
df_train = pd.get_dummies(data=df_train, drop_first=True, columns=categ.columns)

for i in ['RoofMatl_Roll', 'Exterior1st_ImStucc', 'Exterior1st_Stone', 'GarageYrBlt_1914.0', 'Heating_OthW',
          'Condition2_RRAn', 'Heating_GasA', 'GarageQual_Fa', 'GarageYrBlt_1929.0', 'RoofMatl_Metal', 'Electrical_Mix',
          'Utilities_NoSeWa', 'HouseStyle_2.5Fin', 'GarageYrBlt_1906.0', 'Exterior2nd_Other', 'GarageYrBlt_1933.0',
          'RoofMatl_Membran', 'GarageYrBlt_1908.0', 'GarageYrBlt_1931.0', 'Condition2_RRNn',
          'Exterior1st_CBlock']:
    df_train = df_train.drop(i, axis=1)

# Разделение датасета на признаки и целевую переменную
data_train = df_train.copy()
y_train = data_train['SalePrice'].copy()
data_train = data_train.drop('SalePrice', axis=1)





# ИСПЫТАТЕЛЬНЫЙ НАБОР ДАННЫХ

df_test = pd.read_csv('test.csv')
y_test = pd.read_csv('sample_submission.csv')
y_test = y_test.drop(
    [209, 992, 1150, 660, 1116, 95, 1029, 691, 728, 455, 485, 756, 1013, 790, 1444, 757, 758, 27, 888, 580, 725, 1064,
     666])

# Очистка выбросов
df_test = df_test.drop([209, 992, 1150], axis=0)  # MasVnrType
df_test = df_test.drop(
    [660, 1116, 95, 1029, 691, 728, 455, 485, 756, 1013, 790, 1444, 757, 758, 27, 888, 580, 725, 1064, 666], axis=0)

# Работа с отсутствующими данными
df_test = df_test.drop('Id', axis=1)
df_test = df_test.drop('PoolQC', axis=1)
df_test = df_test.drop('MiscFeature', axis=1)
df_test = df_test.drop('Alley', axis=1)
df_test = df_test.drop('Fence', axis=1)
df_test['MasVnrType'] = df_test['MasVnrType'].fillna('None')
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna('None')
df_test['LotFrontage'] = df_test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda value: value.fillna(value.mean()))
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)
df_test['BsmtQual'] = df_test['BsmtQual'].fillna('None')
df_test['BsmtCond'] = df_test['BsmtCond'].fillna('None')
df_test['BsmtExposure'] = df_test['BsmtExposure'].fillna('None')
df_test['BsmtFinType1'] = df_test['BsmtFinType1'].fillna('None')
df_test['BsmtFinType2'] = df_test['BsmtFinType2'].fillna('None')

df_test['GarageType'] = df_test['GarageType'].fillna('None')
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna('None')
df_test['GarageFinish'] = df_test['GarageFinish'].fillna('None')
df_test['GarageQual'] = df_test['GarageQual'].fillna('None')
df_test['GarageCond'] = df_test['GarageCond'].fillna('None')

# Масштабирование
df_test['MiscVal'] = (df_test['MiscVal'] - df_test['MiscVal'].min()) / (
            df_test['MiscVal'].max() - df_test['MiscVal'].min())

# Работа с категориальными атрибутами
df_test['MSSubClass'] = df_test['MSSubClass'].apply(str)
categ = df_test.select_dtypes(include='object')
df_test = pd.get_dummies(data=df_test, drop_first=True, columns=categ.columns)
data_test = df_test.copy()
array = []


for i in list({'GarageYrBlt_1900.0', 'MSSubClass_150', 'Condition2_PosN', 'GarageYrBlt_1943.0', 'GarageYrBlt_1919.0', 'GarageYrBlt_2207.0', 'GarageYrBlt_1917.0'}):
    data_test = data_test.drop(i, axis=1)

# print(data_train)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_error
#
y_test = y_test.drop('Id', axis=1)
reg = LinearRegression()
#
reg.fit(data_train, y_train)
predict = reg.predict(data_test)
predict_serias = pd.Series(predict)
y_seris = pd.Series(y_test['SalePrice']).reset_index()
y_seris = y_seris.drop('index', axis=1)
ost_reg = y_seris['SalePrice'] - predict_serias
# plt.scatter(data_test['GrLivArea'], predict, color='r')
# plt.scatter(data_test['GrLivArea'], y_test['SalePrice'])
# plt.show()

print(np.sqrt(mean_squared_error(y_test, predict)))
#
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()

tree.fit(data_train, y_train)
predict_tree = tree.predict(data_test)
print(np.sqrt(mean_squared_error(y_test, predict_tree)))
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()
forest.fit(data_train, y_train)
predict_forest = forest.predict(data_test)
print(np.sqrt(mean_squared_error(y_test, predict_forest)))
# plt.scatter(data_test['GrLivArea'], predict_forest, color='r')
# plt.scatter(data_test['GrLivArea'], y_test['SalePrice'])
# plt.show()