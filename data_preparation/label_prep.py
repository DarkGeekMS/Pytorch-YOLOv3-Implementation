import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, transform
from scipy.misc import imsave


def label_prep(train_path, test_path, train_labels_path, sub_path):

    train_labels = pd.read_csv(train_labels_path)
    sub = pd.read_csv(sub_path)

    train_no_nan = train_labels[train_labels.notnull().all(axis = 1)]
    train_nan = train_labels[train_labels.isnull().any(axis = 1)]

    train_no_nan['x'] = train_no_nan['x'] + (train_no_nan['width'] / 2)
    train_no_nan['y'] = train_no_nan['y'] + (train_no_nan['height'] / 2)

    train_labels_new = pd.concat([train_no_nan, train_nan])

    len(train_labels_new)

    print(train_labels_new['patientId'].value_counts().shape[0])

    target_column = train_labels_new["Target"]
    train_labels_new = train_labels_new.drop("Target", 1)

    train_labels_new.insert(1, "Target", target_column)

    img_list = os.listdir(train_path)

    img_list

    new_img_list = []
    for name in img_list:
        new_name = name[:-4]
        new_img_list.append(new_name)

    new_img_list

    for img in new_img_list:

        f= open(train_labels_path + "/{}.txt".format(img),"w+")
        data = train_labels_new.loc[train_labels_new['patientId'] == img[:-4]]
        data = data.drop("patientId", 1)
        data = data.as_matrix()
        data = data.reshape((-1, 1))

        if data[0] != 0:
            for i in range(len(data)):
                if data[i] == 1.0:
                    f.write(str(0) + " ")
                else:
                    f.write(str(float(data[i])) + " ")

        f.close()

    train_img_list = os.listdir(train_path + "/train")

    paths_list = []
    for name in train_img_list:
        paths_list.append(train_path + "/train/{}".format(name))

    paths_list

    len(paths_list)

    f= open("train_imgs_list.txt","w+")
    for name in paths_list:
        f.write("\n" + name)
    f.close()

    val_img_list = os.listdir(train_path + "/val")

    paths_list = []
    for name in val_img_list:
        paths_list.append(train_path + "/val/{}".format(name))

    paths_list

    len(paths_list)

    f= open("val_imgs_list.txt","w+")
    for name in paths_list:
        f.write("\n" + name)
    f.close()

    test_img_list = os.listdir(test_path)

    paths_list = []
    for name in test_img_list:
        paths_list.append(test_path + "/{}".format(name))

    paths_list

    len(paths_list)

    f= open("test_imgs_list.txt","w+")
    for name in paths_list:
        f.write("\n" + name)
    f.close()
