#!/usr/bin/env python    
#coding=utf-8    

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy.io as scio 
import numpy as np
import matplotlib.pyplot as plt  
import h5py

#load data
mat_train=u'/home/clarence/Desktop/OceanMiner/Training/training_data_ocean_miner_48_48_sobel.mat'  
train_data=scio.loadmat(mat_train)  

mat_val=u'/home/clarence/Desktop/OceanMiner/Testing/testing_data_ocean_miner_48_48_sobel.mat'  
val_data=scio.loadmat(mat_val) 

plt.close('all')  
x_train=train_data['image']  
y_train=train_data['label']
x_test=val_data['image']
y_test=val_data['label'] 

print 'train original shape:', x_train.shape
print "data loaded successfully!"  

#set training parameters 
batch_size = 40
num_classes = 2
epochs = 30

# input image dimensions
img_rows, img_cols = 48, 48

if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics= ['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('/home/clarence/Desktop/OceanMiner/Model/ocean_miner_48_48_model.h5') 