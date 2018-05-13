import scipy.io as scio 
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

#load data
mat_train=u'/home/clarence/Desktop/OceanMiner/Training/training_data_ocean_miner_48_48_gray.mat'  
train_data=scio.loadmat(mat_train)  

mat_val=u'/home/clarence/Desktop/OceanMiner/Testing/testing_data_ocean_miner_48_48_gray.mat'  
val_data=scio.loadmat(mat_val) 

plt.close('all')  
x_train=train_data['image']  
y_train=train_data['label']
x_test=val_data['image']
y_test=val_data['label'] 

y_train = y_train.ravel()
y_test = y_test.ravel()

print y_train
print 'train original shape:', x_train.shape
print "data loaded successfully!"  

clf = LinearDiscriminantAnalysis()
clf.fit(x_train, y_train)
print(clf.predict(x_train[100]))

joblib.dump(clf, '/home/clarence/Desktop/OceanMiner/Model/Lda_Model.pkl')
