from keras.models import load_model
import cv2
import os
import numpy as np
# from keras.preprocessing import image
# from keras.applications.inception_v3 import preprocess_input
import scipy.io as scio
import matplotlib.pyplot as plt

def case_insensitive_sort(liststring):
    listtemp = [(x.lower(), x) for x in liststring]
    listtemp.sort()
    return [x[1] for x in listtemp]


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):

        print "Scan started!"
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''
            counter = 0
            list = []
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    counter += 1
                    list.append(os.path.join(dirpath, special_file))
                    # files_list.append(os.path.join(dirpath,special_file))
                    # print counter

            if counter > 2:
                print "Found ", counter, " files"
                files_list.extend(list)

        files_list = case_insensitive_sort(files_list)

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


if __name__ == "__main__":

    model = load_model('model_224_224_resnet.h5')

    image_row = 224
    image_col = 224

    #dir = "/home/ubuntu/powerLine/0803rjjpic/roundB/ROI_100"
    dir = "/home/ubuntu/powerLine/ValidationData/0"
    scan = ScanFile(dir)
    # subdirs=scan.scan_subdir()
    files = scan.scan_files()

    counter_normal = 0.0
    counter_abnormal = 0.0

    ### For test
    # mat_train = u'training_data_bgr_224_224_3.mat'
    # train_data = scio.loadmat(mat_train)
    #
    # plt.close('all')
    # x_train = train_data['image']
    #
    # x_train = x_train.astype('float32')
    # x_train /= 255.0
    #
    # value = model.predict(x_train)
    # for v in value:
    #     print v
    ### End of test

    for file in files:
        if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
            img = cv2.imread(file)
            img_standard = cv2.resize(img, (image_row, image_col), interpolation=cv2.INTER_CUBIC)
            #img_predict = img_standard.reshape(1, image_row, image_col, 3)
            img_predict=[]
            img_predict.append(img_standard)
            img_predict = np.array(img_predict)
            img_predict = img_predict.astype('float32')
            img_predict /= 255.0

            value = model.predict(img_predict) #, batch_size=32, verbose=1)

            print value
            if value[0][0] > value[0][1]:
                counter_normal += 1.0
            else:
                counter_abnormal += 1.0

    print "Normal Rate: " + str(counter_normal / (counter_normal + counter_abnormal))
    print "Abnormal Rate: " + str(counter_abnormal / (counter_normal + counter_abnormal))
