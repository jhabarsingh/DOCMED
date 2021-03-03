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

classes = ['covid', 'normal', 'others', 'viral']
train_df = pd.DataFrame(columns=['image','clas'])
val_df = pd.DataFrame(columns=['image','clas'])

for label in classes:
    images = f'./mini_natural_images/{label}'
    print(images)
    for image in os.listdir(images)[:-30]:
      train_df = train_df.append({'image':'./mini_natural_images/'+label+'/'+image,'clas':label},ignore_index=True)
    for image in os.listdir(images)[-30:]:
      val_df = val_df.append({'image':'./mini_natural_images/'+label+'/'+image,'clas':label},ignore_index=True)

print(train_df)


val_df.head()

train_df.shape,val_df.shape

train_datagen = ImageDataGenerator(
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    rescale=1/255,
)

val_datagen = ImageDataGenerator(
    rescale=1/255
)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='image',y_col='clas',classes=classes)
val_generator = val_datagen.flow_from_dataframe(val_df,x_col='image',y_col='clas',classes=classes)

from keras.applications.inception_resnet_v2 import InceptionResNetV2

inceptionresnet = InceptionResNetV2(include_top=False, input_shape=(256,256,3),classes=4)

inceptionresnet.trainable = False

last_layer = inceptionresnet.layers[-1].output

x = GlobalAveragePooling2D()(last_layer)
x = Dense(4,activation='softmax')(x)

model = Model(inceptionresnet.inputs,x)

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),metrics=['acc'])

history = model.fit(train_generator,epochs=3,validation_data=val_generator)


model.save("pickle.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

index = randint(1,val_df.shape[0])
image = val_df.iloc[index]
img = load_img(image.image,target_size=(256,256))
plt.imshow(img)

img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

prediction = model.predict(img_tensor)
classes[np.argmax(prediction)]
