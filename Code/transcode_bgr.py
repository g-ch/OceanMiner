import os
import cv2
import scipy.io as scio
import numpy as np


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


    dir = "TrainingData"
    save_fn = 'training_data_bgr_224_224_3.mat'


    scan = ScanFile(dir)
    # subdirs=scan.scan_subdir()
    files = scan.scan_files()
    # print files

    # get numbers of .jpg
    files_num = 0
    for file in files:
        if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
            files_num += 1

    print files_num

    # cv2.namedWindow("Image")

    image_row = 224
    image_col = 224

    data_array = np.ones((files_num, image_row * image_col * 3), dtype=np.int16)
    label_array = np.ones((files_num, 1), dtype=np.int16)

    counter = 0

    for file in files:
        if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':

            img = cv2.imread(file, 1)
            img_standard = cv2.resize(img, (image_row, image_col), interpolation=cv2.INTER_CUBIC)

            # split the last subdirectory name to use as label(number start from 0)
            label = file.split('/')[-2].split('/')[-1]

            # change label to number
            label_array[counter, 0] = int(label)

            for i in range(image_row):
                for j in range(image_col):
                    data_array[counter, (i * image_col + j) * 3] = img_standard[j, i, 0]
                    data_array[counter, (i * image_col + j) * 3 + 1] = img_standard[j, i, 1]
                    data_array[counter, (i * image_col + j) * 3 + 2] = img_standard[j, i, 2]

            counter += 1

            print str(counter) + "\t" + label

            cv2.imshow("Image", img_standard)
            cv2.waitKey(5)

    scio.savemat(save_fn, {'image': data_array, 'label': label_array})  # save as mat file with two variables
