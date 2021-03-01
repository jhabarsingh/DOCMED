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
import pickle
from keras.models import load_model
from io import 
import h5py


labels = ['covid', 'noncovid']
img_size = 224
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

#Now we can easily fetch our train and validation data.
train = get_data('./input/train')
val = get_data('./input/test')

l = []
for i in train:
    if(i[1] == 0):
        l.append("covid")
    else:
        l.append("noncovid")

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)



model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))

model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])



model.fit(x_train,y_train,epochs = 10 , validation_data = (x_val, y_val))

"""
with h5py.File('pickle.h5', driver='core', backing_store=False) as h5file:
    model.save(h5file)
    h5file.flush()
    serialized = h5file.id.get_file_image().hex()
"""
model.save('pickle.h5')

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

imageName = "noncovid.jpg"
data = readImage(imageName)
predictions = model.predict_classes(data)

if predictions[0] == 0:
    print("COVID")
else:
    print("NONCOVID")


imageName = "covid.png"
data = readImage(imageName)
predictions = model.predict_classes(data)

if predictions[0] == 0:
    print("COVID")
else:
    print("NONCOVID")
