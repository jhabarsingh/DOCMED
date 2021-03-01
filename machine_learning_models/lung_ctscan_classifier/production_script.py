import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
from keras.models import load_model


img_size = 224

# Loading Pickeled Model
model = load_model('pickle.h5')

def readImage(image):
    image = cv2.imread(image)
    resized_arr = cv2.resize(image, (img_size, img_size))
    data = []
    data.append([resized_arr, "noncovid"])
    data = np.array(data)
    train = []
    for feature, label in data:
      train.append(feature)
    data = np.array(train) / 255
    return data


def classify_xray(image):
    imageName = image
    data = readImage(imageName)
    predictions = model.predict_classes(data)
    if(predictions[0] == 0)
        return "covid"
    return "nocovid"



# Testing
# print(classify_xray("noncovid.jpg"))
# print(classify_xray("covid.jpg"))

