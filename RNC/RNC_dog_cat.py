# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:59:05 2024
Treballarem amb una estructura de carpetes
-Dataset
----Single_prediction
--------cat_or_dog.png
-------- ...
----test_set
--------cats
------------cats.png

--------dogs
------------dogs.png

----training_set
--------cats
------------cats.png

--------dogs
------------dogs.png

@author: tvive
"""

# PART 1 - Construir el model de CNN

#Importació de llibreries i paquets
import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Inicialitzar la CNN
classifier = Sequential()

# PAS 1 - Convolució
#filters -->32 detectors de característiques que apliquem
#3,3 --> finestres de 3x3 del detector de característiques
#input_shape --> dimensió de la imatge i canals de color
classifier.add(Conv2D(32, kernel_size = (3,3), input_shape = (64,64,3), activation = "relu"))

# PAS 2 - Max Pooling
# pool_size --> finestra de max pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))


#PAS 3 - Flattening
classifier.add(Flatten())


#PAS 4 - Full Connection
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 1, activation="sigmoid"))


# Compilar la CNN

classifier.compile(optimizer="adam",loss = "binary_crossentropy", metrics = ["accuracy"])












