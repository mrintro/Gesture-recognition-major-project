import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import os


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


print(x_train.shape, x_test.shape)

IMG_Size = 28
x_trainr = np.array(x_train).reshape(-1, IMG_Size, IMG_Size, 1)
x_testr = np.array(x_test).reshape(-1, IMG_Size, IMG_Size, 1)

print(x_trainr.shape, x_testr.shape)

# Neural Network Creation
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (3,3), input_shape= x_trainr.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3,3), input_shape= x_trainr.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3nd convolution layer
model.add(Conv2D(64, (3,3), input_shape= x_trainr.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# fully connected layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# 2nd fully connected layer
model.add(Dense(64))
model.add(Activation('relu'))

# last fully connected layer
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_trainr, y_train, epochs=5, validation_split=0.3)

filename = "cnn_digit_recognizer.h5"

model.save(filename)
