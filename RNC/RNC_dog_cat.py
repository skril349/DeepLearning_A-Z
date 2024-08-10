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

#Repetim per millorar l'accuracy afegint una segona capa de Convulució i max pooling
classifier.add(Conv2D(32, kernel_size = (3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Repetim per millorar l'accuracy afegint una segona capa de Convulució i max pooling
classifier.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))


#PAS 3 - Flattening
classifier.add(Flatten())


#PAS 4 - Full Connection
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 1, activation="sigmoid"))


# Compilar la CNN

classifier.compile(optimizer="adam",loss = "binary_crossentropy", metrics = ["accuracy"])


# PART 2 - Ajustar la CNN a les imatges de train

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 32

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory(
        "dataset/training_set",
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

test_dataset = test_datagen.flow_from_directory(
        "dataset/test_set",
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='binary')

# fine-tune the model
classifier.fit(
        training_dataset,
        steps_per_epoch=8000,#numero de imatges de entrenament
        epochs=25,
        validation_data=test_dataset,
        validation_steps=2000)





#PREDICCIÓ

import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset/single_prediction/cat_or_dog_2.jpg",target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"


