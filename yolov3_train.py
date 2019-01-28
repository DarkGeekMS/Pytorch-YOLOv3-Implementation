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

from skimage.transform import resize

from utilities.helper_func import load_classes, parse_data_config, parse_model_config, weights_init_normal
from yolov3_model import EmptyLayer, YOLOLayer, Darknet
from utilities.data_classes import ImageFolder, ListDataset
from utilities.yolov3_utils import compute_ap, bbox_iou, non_max_suppression, build_targets, to_categorical, create_modules

train_arg = dict()

train_arg["epochs"] = 50  #Change the epochs count according to your need
train_arg["batch_size"] = 4
train_arg["model_config_path"] = "configs/yolov3.cfg"
train_arg["data_config_path"] = "configs/yolov3.data"
train_arg["class_path"] = "configs/yolov3.names"
train_arg["conf_thres"] = 0.8
train_arg["nms_thres"] = 0.4
train_arg["n_cpu"] = 0
train_arg["img_size"] = 512
train_arg["checkpoint_interval"] = 1
train_arg["checkpoint_dir"] = "checkpoints"
train_arg["use_cuda"] = True

cuda = torch.cuda.is_available() and train_arg["use_cuda"]

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Get classes names
classes = load_classes(train_arg["class_path"])

# Get data configuration
data_config     = parse_data_config(train_arg["data_config_path"])
train_path      = data_config['train']

# Get hyper parameters
hyperparams     = parse_model_config(train_arg["model_config_path"])[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
model = Darknet(train_arg["model_config_path"])
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path),
    batch_size=train_arg["batch_size"], shuffle=False, num_workers=train_arg["n_cpu"])

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

for epoch in range(train_arg["epochs"]):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()

        optimizer.step()

        print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                    (epoch, train_arg["epochs"], batch_i, len(dataloader),
                                    model.losses['x'], model.losses['y'], model.losses['w'],
                                    model.losses['h'], model.losses['conf'], model.losses['cls'],
                                    loss.item(), model.losses['recall']))

        model.seen += imgs.size(0)

    if epoch % train_arg["checkpoint_interval"] == 0:
        model.save_weights('%s/%d.weights' % (train_arg["checkpoint_dir"], epoch))
