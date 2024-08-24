# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:13:54 2024

Redes neuronales recurrentes RNR
Prediccions de les accions de google

@author: tvive
"""

#Parte1 - Preprocesado de los datos

#Importacion de librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importem dataset d'entrenament

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values# Open value

#Escalem característiques
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))

training_set_scaled = sc.fit_transform(training_set)

#Crearem una estructura de dades de 60 timesteps i 1 sortida,
#Serem capaços de mirar 60 dies abans de predir el dia de demà
X_train = []
y_train = []

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])#Agafem de 0 a 59 de la columa 0
    y_train.append(training_set_scaled[i,0])#La prediccio serà el dia 60 de la columa 0
    
X_train, y_train = np.array(X_train),np.array(y_train)

# Redimensionament de les dades
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))


#Parte2 - Construccion de la RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

#Inicialització del model
regressor = Sequential()
# Afegim la primera capa de LSTM i la regularització per dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# Afegim la segona capa de LSTM i la regularització per dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Afegim la tercera capa de LSTM i la regularització per dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Afegim la quarta capa de LSTM i la regularització per dropout
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))


#Afegim la capa Densa per donar la sortida
regressor.add(Dense(units = 1))


#Compilar la Xarxa neuronal recurrent

regressor.compile(optimizer = "adam", loss = "mean_squared_error")


# Ajustem la RNR al nostre conjunt de dades d'entrenament

regressor.fit(X_train,y_train, epochs = 100, batch_size = 32 )


#Parte3 - ajustar pesos y visualizar resultados
#Obtenir el valor real de les accions del gener de 2017

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values# Open value

#Predir les accions del gener de 2017 amb la RNR
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]),axis=0)#axis = 0 , serveix per fila mantenint les columnes
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
#necessitem fer un reshape de inputs per fer un vector de una columna
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])#Agafem de 0 a 59 de la columa 0
    
X_test = np.array(X_test)
# Redimensionament de les dades
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)
predict_stock_price_real_value = sc.inverse_transform(predicted_stock_price)

# Visualitzar els resultats obtinguts

plt.plot(real_stock_price,color = "red",label = "Precio Real de la acción de Google")
plt.plot(predict_stock_price_real_value, color = "blue", label = "Precio predecido de la acción de Google")
plt.title("Prediccion con RNR del valor de las acciones de Google")
plt.xlabel("Fecha")
plt.ylabel("precio de la acción de google ($)")
plt.legend()
plt.show()





















