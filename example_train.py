#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_train
# Description : Example of YOLOv5 training using the yolov5utils
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 30-December-2022 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import yolov5

###############################################################################
# PARAMETERS
###############################################################################

# Path to the dataset folder containing images and labels subfolders.
PATH_DATASET='../../../DATA/BURROWS/DATASET/'

# Training image width
IMG_WIDTH=640

# Existing classes in the dataset
CLASS_NAMES=['BURROWS']

# Number of epochs to train
TRAIN_EPOCHS=1

# Path to the projects folder
PATH_PROJECT='PROJECTS'

# Name of the project
NAME_PROJECT='BDETECT'

###############################################################################
# MAIN CODE
###############################################################################

# Install YOLOv5
pathYOLOv5=yolov5.install()

# Build the dataset YAML file
yamlFileName=yolov5.build_dataset_yaml_file(PATH_DATASET,pathYOLOv5,CLASS_NAMES)

# Train
pathModel=yolov5.train(pathYOLOv5=pathYOLOv5,
                       imgWidth=IMG_WIDTH,
                       initialWeights=yolov5.WEIGHTS_SMALL,
                       yamlDataSet=yamlFileName,
                       nEpochs=TRAIN_EPOCHS,
                       pathProject=PATH_PROJECT,
                       nameProject=NAME_PROJECT,
                       theCache=yolov5.CACHE_RAM,
                       forceCPU=True,
                       numWorkers=15)