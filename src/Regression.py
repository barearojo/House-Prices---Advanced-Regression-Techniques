#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:29:04 2023

@author: juan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
import lightgbm as lgb
from StackingAveragedModels import StackingAveragedModels


#leemos los datos tanto del entrenamiento como del test y marcamos lo perdidos como NaN
train = pd.read_csv("../data/train.csv") # Definimos na_values para identificar bien los valores perdidos
test = pd.read_csv("../data/test.csv")

#quitamos dos outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#quitamos las columna id pero la guardamos para luego
if 'Id' in train:
    train.drop('Id', axis=1, inplace=True)
    
test_ids = test.Id
test = test.drop('Id', axis=1)

#Concateno la entrada de ambos para los procesos de etiquetado, que aprenda con ambos conjuntos
input_all = pd.concat([train.drop('SalePrice', axis=1), test])

#elimino columnas con demasiados valores perdidos
input_all["PoolQC"] = input_all["PoolQC"].fillna("None")
input_all["MiscFeature"] = input_all["MiscFeature"].fillna("None")
input_all["Alley"] = input_all["Alley"].fillna("None")
input_all["Fence"] = input_all["Fence"].fillna("None")
input_all["FireplaceQu"] = input_all["FireplaceQu"].fillna("None")
input_all["BsmtQual"] = input_all["BsmtQual"].fillna("BsmtQual")
input_all["BsmtCond"] = input_all["BsmtCond"].fillna("BsmtQual")
input_all["BsmtExposure"] = input_all["BsmtExposure"].fillna("BsmtQual")
input_all["BsmtFinType1"] = input_all["BsmtFinType1"].fillna("BsmtQual")
input_all["BsmtFinType2"] = input_all["BsmtFinType2"].fillna("BsmtQual")



col_cat = list(input_all.select_dtypes(exclude=np.number).columns)
#Voy a reemplazar los valores categóricos por el más frecuente 
imputer_cat = SimpleImputer(strategy="most_frequent")
imputer_cat.fit(input_all[col_cat])
train[col_cat] = imputer_cat.transform(train[col_cat])
test[col_cat] = imputer_cat.transform(test[col_cat])


#Ahora reemplazo los valores numéricos por prediccion knni
col_num = list(train.select_dtypes(include=np.number).columns)
col_num.remove('SalePrice')
imputer_num = KNNImputer(n_neighbors=15)
imputer_num.fit(input_all[col_num])
train[col_num] = imputer_num.transform(train[col_num])
test[col_num] = imputer_num.transform(test[col_num])

#Ahora hago el etiquetado con LabelEncoder, usando un diccionario de LabelEncoder
labelers = {}
test_l = test.copy()
train_l = train.copy()

for col in col_cat:
    labelers[col] = LabelEncoder().fit(input_all[col])
    test_l[col] = labelers[col].transform(test[col])
    train_l[col] = labelers[col].transform(train[col])
    
#Defino en X_train los valores sin el atributo a predecir, y. 
y_train = train_l.SalePrice
X_train = train_l.drop('SalePrice', axis=1)

if 'Id' in test_l:
    test_l.drop('Id', axis=1, inplace=True)

X_test = test_l



model_xgb = xgb.XGBRegressor(n_estimators= 5000, max_depth= 4, learning_rate= 0.05, min_child_weight= 1.3, colsample_bytree= 0.5)


model_lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

stacked_averaged_models = StackingAveragedModels(base_models = (model_lgb, model_xgb), meta_model = model_lasso)

values_xgb = cross_val_score(model_xgb, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5)
#values_stack = cross_val_score(stacked_averaged_models, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5)

print("XgBoost resultado ", values_xgb, "\n Media: ", values_xgb.mean())
#print("Stack resultado ", values_stack, "\n Media: ", values_stack.mean())



model_xgb.fit(X_train, y_train)
stacked_averaged_models.fit(X_train, y_train)

pred = model_xgb.predict(X_test)
stack_predict= stacked_averaged_models.predict(X_test)

salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv("../data/resultados.csv", index=False)

salida_stacked = pd.DataFrame({'Id': test_ids, 'SalePrice': stack_predict})
salida_stacked.to_csv("../data/resultados.csv", index=False)



