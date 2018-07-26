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
import math


UNCLASSIFIED = False
NOISE = 0
if_show = 0

def datetime_toString(dt):
    return dt.strftime("%Y-%m-%d-%H")


""" Cluster Code: https://www.cnblogs.com/wsine/p/5180778.html """
def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())

def eps_neighbor(a, b, eps):
    return dist(a, b) < eps

def region_query(data, pointId, eps):
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds

def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    seeds = region_query(data, pointId, eps)
    if len(seeds) < minPts: 
        clusterResult[pointId] = NOISE
        return False
    else:
        clusterResult[pointId] = clusterId 
        for seedId in seeds:
            clusterResult[seedId] = clusterId

        while len(seeds) > 0: 
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            seeds = seeds[1:]
        return True

def dbscan(data, eps, minPts):
    clusterId = 1
    nPoints = data.shape[1]
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1




class Detect(object): 
    """docstring for  detect"""
    def __init__(self):
        super(Detect, self).__init__()

    def draw_rects(self, positions, img):
        times = len(positions)
        sys_time = datetime.datetime.now()

        for m in range(times):
            size = len(positions[m])
            for i in range(size):  
                cv2.rectangle(img, (positions[m][i][0],positions[m][i][1]), (positions[m][i][0]+positions[m][i][2],positions[m][i][1]+positions[m][i][3]), (0,255,0), 2)


    def cluster_draw_rects(self, positions, img): #positions[x][y][w][h]
        times = len(positions)
        points = []
        
        for m in range(times):
            size = len(positions[m])
            for i in range(size):  
                points.append([positions[m][i][0]+positions[m][i][2]/2,positions[m][i][1]+positions[m][i][3]/2])
       

        data = np.mat(points).transpose()
        print data
        clusters, clusterNum = dbscan(data, 80, 2)
        print clusters
        
        check_seq = 1 #NOTE: CLuster 0 is useless
        max_num = 0
        max_seq = 0

        """Find cluster with largest number of points"""
        while check_seq < 100:
            
            num = 0

            for i in range(len(clusters)):
                if clusters[i] == check_seq:
                    num = num + 1

            if num == 0:
                break

            if num > max_num:
                max_num = num
                max_seq = check_seq

            check_seq = check_seq + 1

        """ Add points """
        x=[]; y=[]
        for i in range(len(clusters)):
            if clusters[i] == max_seq:
                x.append(points[i][0])
                y.append(points[i][1])
                cv2.rectangle(img, (positions[m][i][0],positions[m][i][1]), (positions[m][i][0]+positions[m][i][2],positions[m][i][1]+positions[m][i][3]), (0,255,0), 2)

            else:
                cv2.rectangle(img, (positions[m][i][0],positions[m][i][1]), (positions[m][i][0]+positions[m][i][2],positions[m][i][1]+positions[m][i][3]), (255,0,0), 2)

        #print x
        #print y

        #line_kb = np.polyfit(x, y, 1)
        #print line_kb

        x_acc = 0; y_acc = 0
        for i in range(len(x)):
            x_acc = x_acc + x[i]
            y_acc = y_acc + y[i]

        x_aver = x_acc / len(x)
        y_aver = y_acc / len(y)

        global if_show
        if if_show == 1:
            kkk = cv2.waitKey(1000)

        """ Draw direction. NOTE: car position is unstable!!! """
        cv2.line(img, (int(x_aver),int(y_aver)), (int(self.image_width/2 + 200),int(self.image_height/2 + 100)), (0,0,255), 3)

        self.img_show = img
        global if_show
        if if_show==1:
            cv2.imshow("img", img)
            kk = cv2.waitKey(5000)



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
                            global if_show
                            if if_show == 1:
                                img_temp = self.org_img.copy()
                                self.draw_rects(positions, img_temp)
                                cv2.imshow("img", img_temp)
                                kk = cv2.waitKey(20)

                        else:
                            k += 1
                    j += 1

            print "left ",num," targets"
            self.target_num = num


    def target_detect(self, src_img):
        starttime = datetime.datetime.now()
        #img = cv2.imread(file,1) 
        img = cv2.resize(src_img,(self.image_width,self.image_height),interpolation=cv2.INTER_CUBIC)
        img_copy = img.copy()
        self.org_img = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        target_position = [];

        lda_count = 0
        cnn_count = 0

        global if_show 
        img_to_show1 = img.copy()
        img_to_show2 = img.copy()

        for i in range(self.it_times):
            scan_height = int(self.image_height*(self.it_scale**i)) #PROBLEM?
            scan_width = int(self.image_width*(self.it_scale**i))
            self.img_scan =cv2.resize(img_gray,(scan_width,scan_height),interpolation=cv2.INTER_CUBIC)
            
            window_x = 0
            window_y = 0

            target_position_temp = []
            window_rsize_x = int(self.window_width/self.it_scale**i)
            window_rsize_y = int(self.window_height/self.it_scale**i)

            while window_y < self.image_height - self.window_height -300: #NOTE: 300 to remove car part image    
                
                while window_x < self.image_width - self.window_width:   
                    
                    #print window_x, window_y
                    roi_img = self.img_scan[window_y : window_y+self.window_height, window_x : window_x+self.window_width]
                    img_sobel = cv2.Sobel(roi_img,cv2.CV_8U,1,0,ksize=3)
                    #cv2.imshow("roi", roi_img)
                    #kk = cv2.waitKey(20)

                    if if_show == 1:
                        img_to_show1 = img_to_show2.copy()  
                        cv2.rectangle(img_to_show1, (window_x, window_y), (window_x + self.window_width, window_y + self.window_height), (0,255,255), 2)
                        cv2.imshow("img", img_to_show1)
                        kk = cv2.waitKey(5)
                    

                    #Cascade Predict, MLP
                    roi_lda = roi_img.ravel()
                    pre_result = self.lda.predict_proba(roi_lda)
                    lda_count += 1
                    #print pre_result
               
                    if pre_result[0][1] > 0.85:
                        #img_predict = roi_img.reshape(1, 1, self.window_width, self.window_height)
                        img_predict = img_sobel.reshape(1, 1, self.window_width, self.window_height)
                        
                        value = self.model.predict(img_predict, batch_size=32, verbose=0)
                        cnn_count += 1
                        #print value
                        if value[0][1] > 0.8:
                            target_position_temp.append([window_x, window_y, window_rsize_x, window_rsize_y])
                            if if_show == 1:
                                cv2.rectangle(img_to_show2, (window_x, window_y), (window_x + self.window_width, window_y + self.window_height), (0,255,0), 2)
                                

                    window_x += self.it_step

                window_y += self.it_step
                window_x = 0                
                

            target_position.append(target_position_temp)

        #print target_position
        
        self.merge_rects(target_position)
        global if_show
        if if_show == 1:
            kkk = cv2.waitKey(2000)

        if self.target_num > 2:
            self.cluster_draw_rects(target_position, img_copy)       
        #draw_and_save_rects(target_position, img_copy)


        #long running
        endtime = datetime.datetime.now()
        print "Time", (endtime - starttime).seconds
        print "lda:", lda_count, " cnn:", cnn_count
        

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
        self.det_Button.clicked.connect(self.stop)    
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
        self.D.it_step = 15
        print "Parameters Initialized"

    def detect(self):
        self.stop_flag = 0

        #file = "/home/clarence/Desktop/OceanMiner/Testing/Moment.jpg"
        #src_img = cv2.imread(file,1)
        cap = cv2.VideoCapture("../test.avi")
        while(1):
            ret, src_img = cap.read() #read 
            self.D.target_detect(src_img) # detect

            cv2ImageRGB = cv2.cvtColor(self.D.img_show, cv2.COLOR_BGR2RGB)
            qimg = QImage(cv2ImageRGB.data, cv2ImageRGB.shape[1], cv2ImageRGB.shape[0], QImage.Format_RGB888)
            self.image_View.setPixmap(QPixmap.fromImage(qimg))

            display_info = "Found " + str(self.D.target_num) 
            self.info_View.setText(display_info)

            if self.stop_flag == 1:  #exit
                break

        #QApplication.processEvents()

    def stop(self):
        self.stop_flag = 1

if __name__=="__main__": 

    app = QApplication(sys.argv)  
    ex = Q_Window()  
    sys.exit(app.exec_())  





