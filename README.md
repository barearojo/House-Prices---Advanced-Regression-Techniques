# House-Prices---Advanced-Regression-Techniques
Solución a la siguiente competición de kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/

# README.md

## Modelado Predictivo para Precios de Viviendas

Este script en Python realiza modelado predictivo para precios de viviendas utilizando varias técnicas de aprendizaje automático. El código aborda la preprocesamiento de datos, eliminación de valores atípicos, imputación de valores faltantes y la construcción de modelos apilados para lograr predicciones precisas. Las principales bibliotecas utilizadas son **pandas, numpy, scikit-learn, xgboost, lightgbm**.

### Dependencias
- Python 3.x
- Bibliotecas necesarias: **pandas, numpy, scikit-learn, xgboost, lightgbm**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

### Carga de Datos

- Lee los datos de entrenamiento y prueba desde archivos CSV (`train.csv` y `test.csv`).

### Eliminación de Outliers

- Elimina los valores atípicos de los datos de entrenamiento según condiciones específicas.

### Preprocesamiento de Datos

- Elimina la columna 'Id' de los datos de entrenamiento.
- Guarda la columna 'Id' de los datos de prueba y la elimina.
- Concatena los datos de entrenamiento y prueba para el etiquetado.
- Maneja los valores faltantes en columnas categóricas con imputaciones específicas.
- Reemplaza los valores faltantes en columnas numéricas utilizando la imputación KNN.
- Utiliza LabelEncoder para transformar variables categóricas en numéricas.

### Modelado

- Inicializa modelos de XGBoost, regresión Lasso y regresión LightGBM.
- Realiza validación cruzada para los modelos XGBoost y LightGBM utilizando el error cuadrático logarítmico medio negativo como métrica de puntuación.

### Entrenamiento

- Entrena los modelos de XGBoost y LightGBM con los datos de entrenamiento preprocesados.

### Predicción

- Genera predicciones utilizando los modelos XGBoost y LightGBM.
- Combina las predicciones usando un promedio ponderado.
- Crea un archivo CSV (`resultados.csv`) que contiene las predicciones finales.






