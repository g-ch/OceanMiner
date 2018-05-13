#!/usr/bin/python

import keras
import numpy as np
import h5py
from keras.models import load_model
import cv2 

model = load_model('/home/clarence/Desktop/OceanMiner/Model/ocean_miner_48_48_model.h5')

image_row = 48
image_col = 48

file = "/home/clarence/Desktop/OceanMiner/Testing/1/17.jpg"
img = cv2.imread(file,1) 
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_standard =cv2.resize(img_gray,(image_row,image_col),interpolation=cv2.INTER_CUBIC)

img_predict = img_standard.reshape(1, 1, image_row, image_col)

value = model.predict(img_predict, batch_size=32, verbose=0)
print len(value[0])
print value





