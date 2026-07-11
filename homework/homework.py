#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
import gzip
import pickle
import glob
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json


def año(fila):
    return 2021-int(fila)

def modelo(x_train,y_train):
#Paso 3
    categoricas = ["Fuel_Type", "Selling_type", "Transmission", "Owner"]
    numericas=[]
    for i in x_train.columns:
        if i not in categoricas:
            numericas.append(i)

    preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("scaler",MinMaxScaler(),numericas,),
    ],
    remainder='passthrough',
)

    pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("SelectKBest", SelectKBest(score_func=f_regression)),
        ("LR", LinearRegression()),
    ]
)

#Paso 4
    parametros= {
        "SelectKBest__k": range(1, 20),
    }
    
    model = GridSearchCV(
    estimator=pipeline,
    param_grid=parametros,
    cv=10,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2
    )
    model.fit(x_train, y_train)

#Paso 5
    if os.path.exists("files/models/"):
        for file in glob.glob(f"files/models/*"):
            os.remove(file)
    else:
        os.makedirs("files/models")

    with gzip.open("files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)


def metricas(x_train, x_test, y_train,y_test, model):
#Paso 6
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "r2": r2_score(y_train, y_train_pred),
    "mse": mean_squared_error(y_train, y_train_pred),
    "mad": mean_absolute_error(y_train, y_train_pred),
    }

    test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "r2": r2_score(y_test, y_test_pred),
    "mse": mean_squared_error(y_test, y_test_pred),
    "mad": mean_absolute_error(y_test, y_test_pred),
    }

    if os.path.exists("files/output/"):
        for file in glob.glob(f"files/output/*"):
            os.remove(file)
    else:
        os.makedirs("files/output")

    with open("files/output/metrics.json", "w") as file:
        file.write(json.dumps(train_metrics) + "\n")
        file.write(json.dumps(test_metrics) + "\n")

#Paso 1
train=pd.read_csv(f'files/input/train_data.csv.zip', compression='zip', index_col=False)
train['Age']=train['Year'].apply(año)
train=train.drop(['Year', 'Car_Name'], axis=1)
train=train.dropna()

test= pd.read_csv(f'files/input/test_data.csv.zip', compression='zip', index_col=False)
test['Age']=test['Year'].apply(año)
test=test.drop(['Year', 'Car_Name'], axis=1)
test=test.dropna()

#Paso 2
x_train=train.drop(columns=['Present_Price'])
y_train=train['Present_Price']

x_test=test.drop(columns=['Present_Price'])
y_test=test['Present_Price']

modelo(x_train,y_train)

with gzip.open("files/models/model.pkl.gz", "rb") as file:
    model = pickle.load(file)

metricas(x_train, x_test, y_train,y_test, model)