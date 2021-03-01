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
model = None

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

def joiner(folder_name, file_name):
    paths = os.path.dirname(os.path.abspath(__file__))
    paths = os.path.dirname(paths)
    paths = os.path.join(paths, 'machine_learning_models')
    paths = os.path.join(paths, folder_name)
    paths = os.path.join(paths, file_name)
    return paths

with open(joiner('chest_xray_classifier', 'pickle.h5'), 'rb') as rfile:
    model = load_model(rfile)

def classify_xray(image):
    imageName = image
    data = readImage(imageName)
    predictions = model.predict_classes(data)
    if predictions[0] == 0:
        return "covid"
    return "nocovid"



# Testing
# print(classify_xray("noncovid.jpg"))
# print(classify_xray("covid.jpg"))

