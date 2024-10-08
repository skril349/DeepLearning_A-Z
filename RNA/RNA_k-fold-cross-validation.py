# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:26:59 2024
Redes neuronales artificiales

1 - install Tensorflow y Keras

conda install -c conda-forge keras

2- Install Theano 

pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git


@author: tvive
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# PAS1 - Preprocessing data categorical data

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# OBLIGATORI ESCALAR LES DADES !!

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# PAS2 - Construir la RNA

# Importar Keras i llibreries adicionals

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Inicialitzem la xarxa neuronal artifical RNA

classifier = Sequential()

# Afegim capes d'entrada i la primera capa oculta

classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu", input_shape=(11,)))

# Afegim una segona capa oculta

classifier.add(Dense(units=6, kernel_initializer="uniform",
                     activation="relu"))

# Afegim la capa final
# SIGMOIDE ens donarà uns valors de probabilitats per la sortida

classifier.add(Dense(units=1, kernel_initializer="uniform",
                     activation="sigmoid"))

# Compilem la xarxa neuronal artificial RNA
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

# Ajustem la RNA al conjunt d'entrenament
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# PAS3 - Evaluar el model i calcular predicción finals

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0][0] + cm[1][1]) / y_pred.shape[0]
print(accuracy)

# Prediccion:
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(new_prediction)
print(new_prediction > 0.5)

# PAS4 - Evaluar la RNA (K-Fold-cross-Validation)

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

def buildClassifier():
    # inicialitzem la xarxa neuronal artifical RNA
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform",
                         activation="relu", input_shape=(11,)))
    classifier.add(Dense(units=6, kernel_initializer="uniform",
                         activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform",
                         activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy",
                       metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(model=buildClassifier, batch_size=10, epochs=100)

# cv = validaciones cruzadas (k-folds)
# n_jobs = -1 es per utilitzar tota la potència del pc (diferents cores)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, verbose=1)


print(f"Accuracy mean: {accuracies.mean()}")
print(f"Accuracy std: {accuracies.std()}")
