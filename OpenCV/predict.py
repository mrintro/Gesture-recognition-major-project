import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os

def load_cnn_model():
    cnn_model = load_model('cnn_digit_recognizer.h5')
    return cnn_model


def predict(cnn_model, image_to_predict):
    print("PREDICTING...")
    print(image_to_predict.shape)
    image_to_predict_gray = cv2.cvtColor(image_to_predict, cv2.COLOR_BGR2GRAY)
    print(image_to_predict_gray.shape)
    image_to_predict_gray_resized = cv2.resize(image_to_predict_gray, (28, 28), interpolation=cv2.INTER_AREA)
    image_to_predict_normalized = tf.keras.utils.normalize(image_to_predict_gray_resized, axis=1)
    image_to_predict_normalized = np.array(image_to_predict_normalized).reshape(-1, 28, 28, 1)
    # cv2.imwrite("Images/new.jpg", image_to_predict_normalized)
    prediction = cnn_model.predict(image_to_predict_normalized)


    return np.argmax(prediction), np.amax(prediction)




