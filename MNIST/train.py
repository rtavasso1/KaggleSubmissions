# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 00:32:40 2022

@author: Riley
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

enc = OneHotEncoder()

# Load Data
df_train = pd.read_csv('train.csv').to_numpy()
df_test = pd.read_csv('test.csv').to_numpy()

# Split and OneHot Label
X_train = df_train[:,1:].reshape(42000,28,28,1).astype('float')
y_train = df_train[:,0].reshape(-1,1)
y_train = enc.fit_transform(y_train).toarray()

X_test = df_test.reshape(28000,28,28,1).astype('float')

# Create Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Build Sequential Model
model = Sequential()
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=3150,
    decay_rate=0.96,
    staircase=True)
opt = optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate
y_pred = model.predict(X_test)
y_pred= np.argmax(y_pred,axis=1)

ImageId = np.array(range(1,len(X_test)+1))
submission = pd.DataFrame(columns=['ImageId','Label'])
submission.ImageId = ImageId
submission.Label = y_pred
submission.to_csv('submission.csv', index=False)

"""
Achieves 99% Accuracy on Kaggle
"""