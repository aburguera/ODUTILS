#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : example_detect
# Description : Example of YOLOv5 inference using the yolov5utils
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 17-February-2023 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import os
import numpy as np
from skimage.io import imread
import yolov5
from odmetrics import plot_labeled_image

###############################################################################
# PARAMETERS
###############################################################################

# Path to the folder containing the images to use
PATH_IMAGES='../../../DATA/BURROWS/DATASET/images/test/'

# Path to the trained model
PATH_MODEL='MODELS/BLARGE.pt'

###############################################################################
# MAIN CODE
###############################################################################

# Install YOLOv5
pathYOLOv5=yolov5.install()

# Detect directly using Torch (instead of detect.py)
theNames,predTimes,predLabels=yolov5.detect_with_torch(pathYOLOv5,PATH_MODEL,PATH_IMAGES)

# Let's show some results to ease understanding the output.
# Print the average prediction time.
print('[INFO] AVERAGE PREDICTION TIME PER IMAGE: %.3f ms'%(np.mean(predTimes)))

# Remove outliers from time consumption (usually the first ones are the largest
# due -probably- to startup, caching and stuff)
theMedian=np.median(predTimes)
theStd=np.std(predTimes)
filteredTimes=[[i,t] for i,t in enumerate(predTimes) if abs(t-theMedian)<=3*theStd]

# Get the largest prediction time within the filtered ones
iLargest=np.argmax([x[1] for x in filteredTimes])
iLargest=filteredTimes[iLargest][0]

# Print some related info
print('[INFO] THE MOST TIME CONSUMING IMAGE WAS %s, WHICH REQUIRED %.3f ms TO DETECT %d OBJECTS.'%(theNames[iLargest],predTimes[iLargest],len(predLabels[iLargest])))

# Load the image responsible for the largest time consumption.
theImage=imread(os.path.join(PATH_IMAGES,theNames[iLargest]+'.png'))

# Plot it together with the labels. Note that, for some reason, PyTorch changes
# the matplitlib backend. It is restored by detect_with_torch but sometimes
# restoring it is not enough. So, sometimes the image may not be shown.
plot_labeled_image(theImage,predLabels[iLargest])