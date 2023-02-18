#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import yolov5
import odmetrics

###############################################################################
# PARAMETERS
###############################################################################

# Path to the folder containing the test images and labels (ground truth)
PATH_TEST_IMAGES='../../../DATA/BURROWS/DATASET/images/test/'
PATH_TEST_LABELS='../../../DATA/BURROWS/DATASET/labels/test/'

# Path to the trained model
PATH_MODEL='MODELS/BSMALL.pt'

# Image size (required to convert YOLOv5 (xywhn) to ABSLUTE (xyxy))
IMG_WIDTH=640
IMG_HEIGHT=480

# IoU thresholds to use when computing metrics.
IOU_THRESHOLDS=np.round(np.arange(0.1,0.95,0.05),2)

# Classes to evaluate. Must be an iterable.
CLASSES_TO_EVALUATE=[0]

###############################################################################
# MAIN CODE
###############################################################################

# =============================================================================
# OBJECT DETECTION
# =============================================================================

# Install YOLOv5
pathYOLOv5=yolov5.install()

# Perform object detection
theNames,predTimes,predLabels=yolov5.detect_with_torch(pathYOLOv5,PATH_MODEL,PATH_TEST_IMAGES)

# =============================================================================
# STATS NOT REQUIRING ODMETRICS
# =============================================================================

# Let's show some results (not requiring odmetrics)
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

# Get the smallest prediction time within the filtered ones
iSmallest=np.argmin([x[1] for x in filteredTimes])
iSmallest=filteredTimes[iSmallest][0]

# Print some related info
print('[INFO] THE LEAST TIME CONSUMING IMAGE WAS %s, WHICH REQUIRED %.3f ms TO DETECT %d OBJECTS.'%(theNames[iSmallest],predTimes[iSmallest],len(predLabels[iSmallest])))

# =============================================================================
# USING ODMETRICS.COMPUTE_METRICS AND RELATED UTILITIES
# =============================================================================

# Load the ground truth labels. Note that "theNames" is necessary to load them
# in the same order than the predLabels.
gtLabels=odmetrics.load_all_yolo_labels(PATH_TEST_LABELS,theNames,IMG_WIDTH,IMG_HEIGHT)

# Compute the main metrics
allResults=odmetrics.compute_metrics(predLabels,gtLabels,IOU_THRESHOLDS,CLASSES_TO_EVALUATE)

# Plot some Recall-Precision curves
odmetrics.plot_rp_curves(allResults, [0.1,0.3,0.5,0.7,0.9])

# Get mAP0.5 and mAP0.5:0.05:0.95, both for class 0.
mAP05=odmetrics.compute_mAP(allResults,[0.5],[0])
mAP05_005_095=odmetrics.compute_mAP(allResults,np.round(np.arange(0.5,0.95,0.05),2),[0])

# Print the results
print('[INFO] HERE ARE SOME STATS FOR YOUR ENJOYMENT:')
print('* mAP@0.5=%.3f'%mAP05)
print('* mAP@0.5:0.05:0.95=%.3f'%mAP05_005_095)

# =============================================================================
# USING ODMETRICS.EXPLORE_PARAMETERS AND RELATED UTILITIES
# =============================================================================

# Let's explore the threshold space
theExploration=odmetrics.explore_parameters(predLabels,gtLabels,np.round(np.arange(0.1,0.9,0.1),2), np.round(np.arange(0.1,0.9,0.1),2), [0])

# Plot the F1-Score for all the explored space
odmetrics.plot_exploration(theExploration,metricToShow=5)

# Search the best threshold combination for class 0 according to the different
# explored metrics.
bestTPConfig=odmetrics.get_best_configuration(theExploration,0)
bestFPConfig=odmetrics.get_best_configuration(theExploration,1)
bestFNConfig=odmetrics.get_best_configuration(theExploration,2)
bestRecallConfig=odmetrics.get_best_configuration(theExploration,3)
bestPrecisionConfig=odmetrics.get_best_configuration(theExploration,4)
bestF1Config=odmetrics.get_best_configuration(theExploration,5)

# Print the configurations
print('[INFO] CONFIGURATION MAXIMIZING TRUE POSITIVES')
odmetrics.print_explored_configuration(bestTPConfig)
print('\n[INFO] CONFIGURATION MAXIMIZING FALSE POSITIVES')
odmetrics.print_explored_configuration(bestFPConfig)
print('\n[INFO] CONFIGURATION MAXIMIZING FALSE NEGATIVES')
odmetrics.print_explored_configuration(bestFNConfig)
print('\n[INFO] CONFIGURATION MAXIMIZING RECALL')
odmetrics.print_explored_configuration(bestRecallConfig)
print('\n[INFO] CONFIGURATION MAXIMIZING PRECISION')
odmetrics.print_explored_configuration(bestPrecisionConfig)
print('\n[INFO] CONFIGURATION MAXIMIZING F1-SCORE')
odmetrics.print_explored_configuration(bestF1Config)