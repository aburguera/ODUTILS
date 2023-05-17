#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import odmetrics
from prettytable import PrettyTable

###############################################################################
# PARAMETERS
###############################################################################

#==============================================================================
# MAIN PARAMETERS
#==============================================================================

# Paths to ground truth and predicted labels.
# * File names must be coincident in both directories.
# * All the images must have one ground truth label file, even if it is empty.
# * Images with no predictions can either have an empty prediction label file
#   or no prediction file at all.
# * Ground truth labels must be CLASS X Y WIDTH HEIGHT (RELATIVE)
# * Predicted labels must be CLASS X Y WIDTH HEIGHT SCORE (RELATIVE)
PATH_GT_LABELS='../../../DATA/BURROWS/GLOBAL_SPLIT/labels/test/'
PATH_PRED_LABELS='PRED_IMG_MEDIUM/labels/'

# Classes to evaluate. Must be an iterable.
CLASSES_TO_EVALUATE=[0,1,2]

# Class names (to print/plot results). Must be in the same order as
# CLASSES_TO_EVALUATE.
CLASS_NAMES=['BURROW','NEPHROP','OTHER']

# Images dimensions. Images are NOT loaded. Dimensions are used only to
# convert labels to absolute coordinates.
IMG_WIDTH=640
IMG_HEIGHT=480

# File extensions.
EXT_GT_LABELS='txt'
EXT_PRED_LABELS='txt'

#==============================================================================
# FINE-TUNING PARAMETERS
#==============================================================================
# No need to modify them. Do not modify them. Seriously, don't.

# IoU values to compute metrics
IOU_METRICS_VALUES=np.round(np.arange(0.05,0.95,0.05),2)
# IoU values to perform parameter exploration. Must exist in IOU_METRICS_VALUES
IOU_EXPLORE_VALUES=np.round(np.arange(0.1,0.9,0.1),2)
# Confidence thresholds to perform parameter exploration.
CONF_EXPLORE_VALUES=np.round(np.arange(0.1,0.9,0.1),2)
# IoU values for which plot the curves (must exist in IOU_METRICS_VALUES)
IOU_PLOT_CURVES=[0.1,0.3,0.5,0.7,0.9]

###############################################################################
# MAIN PROGRAM
###############################################################################

#==============================================================================
# LOAD LABELS
#==============================================================================

# Get the labels file names without extension. Necessary to ensure that
# ground truth and predicted labels are loaded in the same order
theNames=[curFileName[:-4] for curFileName in os.listdir(PATH_GT_LABELS)]
# Load the ground truth labels.
gtLabels=odmetrics.load_all_yolo_labels(PATH_GT_LABELS,theNames,IMG_WIDTH,IMG_HEIGHT,0,EXT_GT_LABELS)
# Load the predicted labels.
predLabels=odmetrics.load_all_yolo_labels(PATH_PRED_LABELS,theNames,IMG_WIDTH,IMG_HEIGHT,0,EXT_PRED_LABELS)

#==============================================================================
# COMPUTE MAIN METRICS
#==============================================================================
# Compute the main metrics
allResults=odmetrics.compute_metrics(predLabels,gtLabels,IOU_METRICS_VALUES,CLASSES_TO_EVALUATE)

# Plot Recall-Precision curves for each class and all together
for curClass in CLASSES_TO_EVALUATE:
    theTitle='CLASS: '+CLASS_NAMES[curClass]
    odmetrics.plot_rp_curves(allResults,IOU_PLOT_CURVES,curClass,theTitle)
odmetrics.plot_rp_curves(allResults,IOU_PLOT_CURVES,CLASSES_TO_EVALUATE,'CLASS: ALL')

# Get mAP0.5 for each class and all together
for curClass in CLASSES_TO_EVALUATE:
    print('[CLASS %s]'%CLASS_NAMES[curClass])
    mAP05=odmetrics.compute_mAP(allResults,[0.5],[curClass])
    mAP05_005_095=odmetrics.compute_mAP(allResults,np.round(np.arange(0.5,0.95,0.05),2),[curClass])
    print('* mAP@0.5=%.3f'%mAP05)
    print('* mAP@0.5:0.05:0.95=%.3f'%mAP05_005_095)
print('[ALL CLASSES]')
mAP05=odmetrics.compute_mAP(allResults,[0.5],CLASSES_TO_EVALUATE)
mAP05_005_095=odmetrics.compute_mAP(allResults,np.round(np.arange(0.5,0.95,0.05),2),CLASSES_TO_EVALUATE)
print('* mAP@0.5=%.3f'%mAP05)
print('* mAP@0.5:0.05:0.95=%.3f'%mAP05_005_095)

#==============================================================================
# COMPUTE ADDITIONAL METRICS
#==============================================================================
# Let's explore the threshold and IoU space space
theExploration=odmetrics.explore_parameters(predLabels,gtLabels,IOU_EXPLORE_VALUES,CONF_EXPLORE_VALUES,CLASSES_TO_EVALUATE)

metricNames=['TRUE POSITIVES','FALSE POSITIVES','FALSE NEGATIVES','RECALL','PRECISION','F1-SCORE']
for iMetric,theName in enumerate(metricNames):
    print('IoU AND CONFIDENCE THRESHOLD OPTIMIZING THE %s'%theName)
    odmetrics.print_exploration_summary(theExploration, CLASSES_TO_EVALUATE, CLASS_NAMES, 5)
    print()

# Now plot the configuration space for all classes (together) and F1-SCORE
# Individual spaces can also be plotted.
odmetrics.plot_exploration_multiclass(theExploration,5,CLASSES_TO_EVALUATE)
plt.title(','.join(CLASS_NAMES))