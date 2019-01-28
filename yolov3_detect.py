from __future__ import division
import math
import time
import datetime
import glob
import random
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from skimage.transform import resize

from utilities.helper_func import load_classes, parse_data_config, parse_model_config, weights_init_normal
from yolov3_model import EmptyLayer, YOLOLayer, Darknet
from utilities.data_classes import ImageFolder, ListDataset
from utilities.yolov3_utils import compute_ap, bbox_iou, non_max_suppression, build_targets, to_categorical, create_modules

test_arg = dict()

test_arg["batch_size"] = 1
test_arg["model_config_path"] = "configs/yolov3.cfg"
train_arg["data_config_path"] = "configs/yolov3.data"
test_arg["class_path"] = "configs/yolov3.names"
test_arg["weights_path"] = "configs/9.weights"
test_arg["conf_thres"] = 0.8
test_arg["nms_thres"] = 0.4
test_arg["n_cpu"] = 0
test_arg["img_size"] = 512
test_arg["use_cuda"] = True

cuda = torch.cuda.is_available() and test_arg["use_cuda"]

# Get data configuration
data_config     = parse_data_config(val_arg["data_config_path"])
test_path       = data_config['test']
num_classes     = int(data_config['classes'])

os.makedirs('output', exist_ok=True)

# Set up model
model = Darknet(test_arg["model_config_path"], img_size=test_arg["img_size"])
model.load_weights(test_arg["weights_path"])

if cuda:
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(test_arg["image_folder"], img_size=test_arg["img_size"]),
                        batch_size=test_arg["batch_size"], shuffle=False, num_workers=test_arg["n_cpu"])

classes = load_classes(test_arg["class_path"]) # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 80, test_arg["conf_thres"], test_arg["nms_thres"])

    print(detections)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)


# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print ('\nSaving images:')
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

    print ("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (test_arg["img_size"] / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (test_arg["img_size"] / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = test_arg["img_size"] - pad_y
    unpad_w = test_arg["img_size"] - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                    edgecolor=color,
                                    facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                    bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()
