#!/usr/bin/python

import keras
import numpy as np
import h5py
from keras.models import load_model
import cv2 
import datetime
def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")

if __name__=="__main__": 

    a = [[2, 3], [3,4]]
    print a 
    del a[1][0]
    print a
    d = datetime.datetime.now()

    a = 5 

    cc = datetime_toString(d) + str(a)
    print cc
