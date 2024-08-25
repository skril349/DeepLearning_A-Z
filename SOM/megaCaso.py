# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:04:56 2024

Combine SOM with ANN

@author: tvive
"""
# Identificar els fraudes potencials amb SOM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importem el dataset

dataset = pd.read_csv("Credit_Card_Applications.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Escalado de característiques
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Entrenar el SOM
from minisom import MiniSom

som = MiniSom(x = 10, y=10, input_len=15,sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X)
som.train_random(data = X, num_iteration=100)

#Visualizar los resultados

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o", "s"]
colors = ["r","g"]
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5, markers[y[i]],markeredgecolor = colors[y[i]], markerfacecolor = "None",
         markersize = 10, markeredgewidth = 2)

show()

# Trobar els fraudes
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(6,8)],mapping[(5,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

# Ara entrenarem la xarxa neuronal amb les variables dependents del SOM

#Parte 2

#Crear la matriu de característiques
customers = dataset.iloc[:,1:-1].values



#Crear la variable dependent
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1



#ANN

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# Feature Scaling
sc_X = StandardScaler()
customers = sc.fit_transform(customers)

#PAS2 - Construir la RNA

#Improtar Keras i llibreries adicionals

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Inicialitzem la xarxa neuronal artifical RNA

classifier = Sequential()

# Afegim capes d'entrada i la primera capa oculta

#Dense -> units s'experimenta. normalment media entre capes d'entrada i sortida 
# entrada = 11, sortida = 1, mitja = 12/2 = 6
# relu -> rectificador lineal unitari
#RELU permet activar les neurones només si son interessants

classifier.add(Dense(units=4, kernel_initializer = "uniform",
                     activation = "relu", input_dim = 14))
classifier.add(Dropout(rate=0.1))

#Afegim una segona capa oculta

classifier.add(Dense(units=6, kernel_initializer = "uniform",
                     activation = "relu"))
classifier.add(Dropout(rate=0.1))

# Afegim la capa final
# SIGMOIDE ens donarà uns valors de probabilitats per la sortida

classifier.add(Dense(units=1, kernel_initializer = "uniform",
                     activation = "sigmoid"))


#Compilem la xarxa neuronal artificial RNA
classifier.compile(optimizer = "adam",loss = "binary_crossentropy" ,
                   metrics=["accuracy"])


# Ajustem la RNA al conjunt d'entrenament
#batch_size -> cada 10 elements, corregeix els pesos
#epochs -> quantes vegades repetim el proces
classifier.fit(customers,is_fraud,batch_size = 1, epochs = 50 )


#PAS3 - Evaluar el model i calcular predicción finals

# Predicting the Test set results
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:,0:1], y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]

