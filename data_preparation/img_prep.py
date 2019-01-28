import pandas as pd
import numpy as np
import os

import pydicom
from pydicom.data import get_testdata_files

import matplotlib.pyplot as plt

from skimage import io, transform
from scipy.misc import imsave

def prep_imgs(train_path, test_path):

    file_list_1 = []
    name_list_1 = []
    for name in os.listdir(train_path):
        file_list_1.append(train_path + "/" + name)
        name_list_1.append(name)
    print("{} images found in train folder".format(len(file_list_1)))

    i = 0
    for img_name in file_list_1:
        img = pydicom.dcmread(img_name).pixel_array
        imsave(train_path + '/{}.jpg'.format(name_list_1[i]), img)
        i += 1

    file_list_2 = []
    name_list_2 = []
    for name in os.listdir(test_path):
        file_list_2.append(test_path + "/" + name)
        name_list_2.append(name)
    print("{} images found in test folder".format(len(file_list_2)))

    i = 0
    for img_name in file_list_2:
        img = pydicom.dcmread(img_name).pixel_array
        imsave(test_path + '/{}.jpg'.format(name_list_2[i]), img)
        i += 1
