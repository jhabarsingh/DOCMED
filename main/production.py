import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import zipfile
import os
from random import randint
from keras.layers import Dense,Flatten,MaxPool2D,Conv2D,Dropout,GlobalAveragePooling2D
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.models import Model,Sequential
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from keras.models import load_model


classes = ['covid', 'normal', 'others', 'viral']

model = load_model('pickle.h5')

def predict(image):
	img = load_img(image,target_size=(256,256))
	img_tensor = img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.
	prediction = model.predict(img_tensor)
	return classes[np.argmax(prediction)]