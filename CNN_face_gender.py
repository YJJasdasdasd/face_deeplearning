import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

dir = '/content/drive/MyDrive/data/all_age_faces'

image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_dataset = image_generator.flow_from_directory(dir,
                                                    classes=['male', 'female'],
                                                    target_size=(300, 300),
                                                    batch_size=20,
                                                    shuffle=True,
                                                    subset='training',
                                                    class_mode='binary')
validation_dataset = image_generator.flow_from_directory(dir,
                                                        classes=[
                                                            'male', 'female'],
                                                        target_size=(
                                                            300, 300),
                                                        batch_size=20,
                                                        shuffle=True,
                                                        subset='validation',
                                                        class_mode='binary')

model = Sequential()
model.add(Conv2D(filters=32,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                input_shape=(300, 300, 3)))
model.add(Conv2D(filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu'))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

modelpath = "./GENDER_CNN.hdf5"
checkpointer = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
