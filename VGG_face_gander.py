import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16
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
model_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(300, 300, 3))
model_base.trainable = False

model.add(model_base)
model.add(Flatten())
model.add(Dense(units=256,
                activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
modelpath = "./GENDER_CNN.hdf5"
checkpointer = ModelCheckpoint(
    filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(train_dataset,
                    steps_per_epoch=100,
                    epochs=30,
                    verbose=1,
                    validation_data=validation_dataset,
                    validation_steps=50,
                    callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(train_dataset)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Treainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
