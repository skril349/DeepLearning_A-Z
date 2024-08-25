# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 18:04:09 2024
Creating SOM AI (Self Organized Maps)
@author: tvive
"""

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

som = MiniSom(x = 10, y=10, input_len=15,sigma=1.0, learning_rate=0.5)
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
frauds = np.concatenate((mapping[(6,1)],mapping[(7,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
