#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:29:04 2023

@author: juan
"""
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

#leemos los datos tanto del entrenamiento como del test y marcamos lo perdidos como NaN
train = pd.read_csv("../data/train.csv", na_values="NaN") # Definimos na_values para identificar bien los valores perdidos
test = pd.read_csv("../data/test.csv", na_values="NaN")

#quitamos las columna id pero la guardamos para luego
if 'Id' in train:
    train.drop('Id', axis=1, inplace=True)
    
test_ids = test.Id
test = test.drop('Id', axis=1)

#Concateno la entrada de ambos para los procesos de etiquetado, que aprenda con ambos conjuntos
input_all = pd.concat([train.drop('SalePrice', axis=1), test])


col_cat = list(input_all.select_dtypes(exclude=np.number).columns)
#Voy a reemplazar los valores categóricos por el más frecuente (es mejorable)
imputer_cat = SimpleImputer(strategy="most_frequent")
imputer_cat.fit(input_all[col_cat])
train[col_cat] = imputer_cat.transform(train[col_cat])
test[col_cat] = imputer_cat.transform(test[col_cat])


#Ahora reemplazo los valores numéricos por prediccion knni
col_num = list(train.select_dtypes(include=np.number).columns)
col_num.remove('SalePrice')
imputer_num = KNNImputer(n_neighbors=5)
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


model = xgb.XGBRegressor()
values = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_log_error', cv=5)
print(values)
print(values.mean())
model.fit(X_train, y_train)
pred = model.predict(X_test)
salida = pd.DataFrame({'Id': test_ids, 'SalePrice': pred})
salida.to_csv("../data/resultados6.csv", index=False)



