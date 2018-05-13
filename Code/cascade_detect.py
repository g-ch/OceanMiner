#!/usr/bin/python

###2018-05-13 Clarence Chen ###

import sys  
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage  

import keras
import numpy as np
import h5py
from keras.models import load_model
import cv2 
import datetime
from sklearn.externals import joblib

def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")

class Detect(object):
    """docstring for  detect"""
    def __init__(self):
        super(Detect, self).__init__()

    def draw_rects(self, positions, img): #positions[x][y][w][h]
        times = len(positions)
        for m in range(times):
            size = len(positions[m])
            for i in range(size):  
                cv2.rectangle(img, (positions[m][i][0],positions[m][i][1]), (positions[m][i][0]+positions[m][i][2],positions[m][i][1]+positions[m][i][3]), (0,255,0), 2)
        self.img_show = img

    def draw_and_save_rects(self, positions, img): #positions[x][y][w][h]
        times = len(positions)
        sys_time = datetime.datetime.now()

        for m in range(times):
            size = len(positions[m])
            for i in range(size):  
                cv2.rectangle(img, (positions[m][i][0],positions[m][i][1]), (positions[m][i][0]+positions[m][i][2],positions[m][i][1]+positions[m][i][3]), (0,255,0), 2)
                roi_img = self.img_scan[positions[m][i][1]: positions[m][i][1]+positions[m][i][3], positions[m][i][0]: positions[m][i][0]+positions[m][i][2]]
                name = datetime_toString(sys_time) + str(i)
                cv2.imwrite('/home/clarence/Desktop/OceanMiner/Result/'+ name + '.jpg',roi_img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
        self.img_show = img


    def merge_rects(self, positions):
        times = len(positions)

        #inside one time, m
        for m in range(times):
            num = len(positions[m])
            print "original ",num," targets"

            if num > 2: #merge for images found more than 2 targets
                j = 0
                window_x_reduce = positions[m][0][2] / 2
                window_y_reduce = positions[m][0][3] / 2
                #compare
                while j < num:
                    k = j + 1
                    #only search nearby area
                    while k < num and (abs(positions[m][j][0] - positions[m][k][0]) < window_x_reduce or abs(positions[m][j][1] - positions[m][k][1]) < window_y_reduce):                
                        if (abs(positions[m][j][1] - positions[m][k][1]) < window_y_reduce) and (abs(positions[m][j][0] - positions[m][k][0]) < window_x_reduce): 
                            #start merge
                            positions[m][j][0] = int((positions[m][j][0] + positions[m][k][0]) / 2)
                            positions[m][j][1] = int((positions[m][j][1] + positions[m][k][1]) / 2)
                            del positions[m][k]
                            num -= 1
                        else:
                            k += 1
                    j += 1

            print "left ",num," targets"


    def target_detect(self, file):
        starttime = datetime.datetime.now()
        img = cv2.imread(file,1) 
        img = cv2.resize(img,(self.image_width,self.image_height),interpolation=cv2.INTER_CUBIC)
        img_copy = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        target_position = [];

        lda_count = 0
        cnn_count = 0

        for i in range(self.it_times):
            scan_height = int(self.image_height*(self.it_scale**i)) #PROBLEM?
            scan_width = int(self.image_width*(self.it_scale**i))
            self.img_scan =cv2.resize(img_gray,(scan_width,scan_height),interpolation=cv2.INTER_CUBIC)
            
            window_x = 0
            window_y = 0

            target_position_temp = []
            window_rsize_x = int(self.window_width/self.it_scale**i)
            window_rsize_y = int(self.window_height/self.it_scale**i)

            while window_y < self.image_height - self.window_height:    
                
                while window_x < self.image_width - self.window_width:   
                    
                    #print window_x, window_y
                    roi_img = self.img_scan[window_y : window_y+self.window_height, window_x : window_x+self.window_width]
                    #cv2.imshow("roi", roi_img)
                    #kk = cv2.waitKey(20)

                    roi_lda = roi_img.ravel()
                    pre_result = self.lda.predict_proba(roi_lda)
                    lda_count += 1
                    #print pre_result

                    #Cascade Predict, MLP
                    if pre_result[0][1] > 0.95:

                        img_predict = roi_img.reshape(1, 1, self.window_width, self.window_height)
                        value = self.model.predict(img_predict, batch_size=32, verbose=0)
                        cnn_count += 1
                        #print value
                        if value[0][1] > 0.8:
                            target_position_temp.append([window_x, window_y, window_rsize_x, window_rsize_y])


                    window_x += self.it_step

                window_y += self.it_step
                window_x = 0                
                

            target_position.append(target_position_temp)

        #print target_position
        
        self.merge_rects(target_position)
        self.draw_rects(target_position, img_copy)
        #draw_and_save_rects(target_position, img_copy)


        #long running
        endtime = datetime.datetime.now()
        print "Time", (endtime - starttime).seconds
        print "lda:", lda_count, " cnn:", cnn_count
        self.target_num = cnn_count

        #cv2.imshow("image", img_copy)
        #kkk = cv2.waitKey()

class Q_Window(QWidget): 
      
    def __init__(self):  
        super(Q_Window, self).__init__()    
        self.initParas()
        self.initUI()  
          
          
    def initUI(self):                 
        
        self.image_View = QLabel("image", self)
        self.image_View.resize(1280, 720)
        self.image_View.setScaledContents(True)
        self.image_View.move(60,30)
        jpg=QPixmap('/home/clarence/Desktop/OceanMiner/Testing/Moment.jpg')  
        self.image_View.setPixmap(jpg) 

        self.info_View = QLabel("Information", self)
        self.info_View.resize(80, 60)
        self.info_View.setScaledContents(True)
        self.info_View.move(1360,440)

        self.det_Button = QPushButton('Detect', self)  
        self.det_Button.clicked.connect(self.detect)  
        self.det_Button.resize(80,40)  #self.det_Button.sizeHint()
        self.det_Button.move(1380, 300)  

        self.det_Button = QPushButton('Stop', self)  
        self.det_Button.resize(80,40)  #self.det_Button.sizeHint()
        self.det_Button.move(1380, 360) 
      
        self.setGeometry(300, 300, 1500, 800)  
        self.setWindowTitle('Miner')      
        self.show() 


    def initParas(self):
        self.D = Detect()

        self.D.model = load_model('/home/clarence/Desktop/OceanMiner/Model/ocean_miner_48_48_model.h5')
        self.D.lda = joblib.load('/home/clarence/Desktop/OceanMiner/Model/Lda_Model.pkl')

        self.D.image_width = 1280
        self.D.image_height = 720

        self.D.window_width = 48
        self.D.window_height = 48

        self.D.it_scale = 1
        self.D.it_times = 1
        self.D.it_step = 10
        print "Parameters Initialized"

    def detect(self):
        file = "/home/clarence/Desktop/OceanMiner/Testing/Moment.jpg"
        self.D.target_detect(file)

        cv2ImageRGB = cv2.cvtColor(self.D.img_show, cv2.COLOR_BGR2RGB)
        qimg = QImage(cv2ImageRGB.data, cv2ImageRGB.shape[1], cv2ImageRGB.shape[0], QImage.Format_RGB888)
        self.image_View.setPixmap(QPixmap.fromImage(qimg))

        display_info = "Found " + str(self.D.target_num) + " Manganese nodules."
        self.info_View.setText(display_info)

        #QApplication.processEvents()


if __name__=="__main__": 

    app = QApplication(sys.argv)  
    ex = Q_Window()  
    sys.exit(app.exec_())  





