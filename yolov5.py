#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : yolov5
# Description : Miscelaneous YOLOv5 utility functions.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 29-December-2022 - Creation
#               17-February-2023 - Major refactor.
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import os,sys
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import matplotlib
from skimage.io import imread
import torch

###############################################################################
# CONSTANTS
###############################################################################

# INITIAL WEIGHTS

WEIGHTS_NANO='yolov5n.pt'
WEIGHTS_SMALL='yolov5s.pt'
WEIGHTS_MEDIUM='yolov5m.pt'
WEIGHTS_LARGE='yolov5l.pt'
WEIGHTS_XLARGE='yolov5x.pt'

# CACHE VALUES

CACHE_RAM='ram'
CACHE_DISK='disk'

# OPTIMIZER VALUES

OPTIMIZER_SGD='SGD'
OPTIMIZER_ADAM='Adam'
OPTIMIZER_ADAMW='AdamW'

###############################################################################
# MAIN FUNCTIONS
###############################################################################

# =============================================================================
# INSTALL
# Installs YOLOv5 by cloning the github repo and installing the requirements.
# It also performs some basic sanity checks.
# Input  : localPath - Where to install YOLOv5 or None to use the repo folder
#                      name as localPath.
#          repoURL - Github repository URL. The default value points to a
#                    fork into my own github just in case. The original one
#                    is https://github.com/ultralytics/yolov5
# Output : localPath - Path where YOLOv5 has been installed.
# =============================================================================
def install(localPath=None,repoURL='https://github.com/aburguera/yolov5'):
    print('[INSTALL] INSTALLING YOLOV5')
    # If no local path is specified, use that of the repo URL.
    if localPath is None:
        localPath=os.path.split(repoURL)[-1]
    # If the path already exists, that's probably because YOLOv5 is already
    # installed.
    if os.path.exists(localPath):
        print('  * PATH %s ALREADY EXISTS. YOLOv5 REPO IS PROBABLY THERE.'%localPath)
    # If the path does not exist...
    else:
        # Clone the repo.
        print('  * CLONING YOLOV5 REPO %s IN %s'%(repoURL,localPath))
        execute_command('git clone %s %s'%(repoURL,localPath))
        # Install requirements.
        print('  * INSTALLING REQUIREMENTS')
        execute_command('pip install -r %s'%os.path.join(localPath,'requirements.txt'))
    print('[INSTALL] DONE')
    # Return the local installation path
    return localPath

# =============================================================================
# TRAIN
# Simple wrapper to execute the YOLOv5 train script.
# Input  : pathYOLOv5 - Path to the local yolov5 installation. Recommended
#                       value is the output of yolov5_install.
#          imgWidth - Images width (512, 640, ...)
#          initialWeights - Initial weights file. Recommended values are
#                           WEIGHTS_NANO, WEIGHTS_SMALL,WEIGHTS_MEDIUM,
#                           WEIGHTS_LARGE and WEIGHTS_XLARGE.
#          yamlDataSet - Dataset specs in the YOLOv5 required YAML format.
#                        to avoid path or format problems, the recommended
#                        value is that returned by
#                        yolov5_build_dataset_yaml_file
#          nEpochs - Number of training epochs.
#          pathProject - Path to the folder where to save weights, ...
#          nameProject - Path to the subfolder where to save weights, ...
#          batchSize - Number of images in a batch. If None, default value is
#                      used.
#          resumeTraining - If None, training is not resumed. If a checkpoint
#                           is specified, training is resumed from there.
#                           If an empty string is specified, YOLOv5 will
#                           guess the appropriate checkpoint.
#          theCache - Possible values are CACHE_DISK and
#                     CACHE_RAM. Using cache suposedly speeds up
#                     training. If None, no cache is used.
#          forceCPU - If True, CPU is used even if GPU is available. If False,
#                     YOLOv5 will select the device (CPU, GPU, ...).
#          theOptimizer - What optimizer use. Possible values are
#                         OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_ADAMW or
#                         None. None means using the default (recommended).
#          numWorkers - If CPU is used, num workers determine the number of
#                       CPU kernels to use. If GPU is used... I'm not sure.
#          pythonCommand - Name of the python interpreter executable.
# Output : pathModel - Path to the trained model weights.
# =============================================================================
def train(pathYOLOv5,imgWidth,initialWeights,yamlDataSet,nEpochs,pathProject,nameProject,batchSize=None,resumeTraining=None,theCache=None,forceCPU=False,theOptimizer=None,numWorkers=None,pythonCommand='python3'):
    # Prepare the command line
    theCommand=pythonCommand+' '+os.path.join(pathYOLOv5,'train.py')+' '
    theCommand+='--img %d '%imgWidth
    theCommand+='--weights %s '%initialWeights
    theCommand+='--data %s '%yamlDataSet
    theCommand+='--epochs %d '%nEpochs
    theCommand+='--project %s '%pathProject
    theCommand+='--name %s '%nameProject
    theCommand+='--exist-ok '
    if not (batchSize is None):
        theCommand+='--batch-size %d '%batchSize
    if not (resumeTraining is None):
        theCommand+='--resume %s '%resumeTraining
    if not (theCache is None):
        theCommand+='--cache %s '%theCache
    if forceCPU:
        theCommand+='--device cpu '
    if not (theOptimizer is None):
        theCommand+='--optimizer %s '%theOptimizer
    if not (numWorkers is None):
        theCommand+='--workers %d'%numWorkers
    # Execute the command line
    print('[TRAINING] TRAINING COMMAND IS %s'%theCommand)
    execute_command(theCommand,hideOutput=False)
    pathModel=os.path.join(pathProject,nameProject,'weights','best.pt')
    print('[TRAINING] FINISHED. WEIGHTS SAVED AT %s'%pathModel)
    return pathModel

# =============================================================================
# DETECT
# Simple wrapper to execute the YOLOv5 detect script.
# Input  : pathYOLOv5 - Path to the local yolov5 installation. Recommended
#                       value is the output of yolov5_install.
#          weightsFile - File with the trained weights to use.
#          inputPath - Path containing the input images.
#          outputPath - Where to save the results
#          saveImages - Save the labeled images (True/False)
#          imgSize - Images width (512, 640, ...)
#          confThreshold - Minimum score to consider an object.
#          forceCPU - If True, CPU is used even if GPU is available. If False,
#                     YOLOv5 will select the device (CPU, GPU, ...).
#          pythonCommand - Name of the python interpreter executable.
# Output : Path to the label files.
# =============================================================================
def detect(pathYOLOv5,weightsFile,inputPath,outputPath,saveImages=False,imgSize=None,confThres=None,forceCPU=False,pythonCommand='python3'):
    # Prepare the command line
    theCommand=pythonCommand+' '+os.path.join(pathYOLOv5,'detect.py')+' --save-txt --save-conf --exist-ok '
    if saveImages==False:
        theCommand+='--nosave '
    theCommand+='--weights %s '%weightsFile
    theCommand+='--source %s '%inputPath
    if not(imgSize is None):
        theCommand+='--img %d '%imgSize
    if not(confThres is None):
        theCommand+='--conf-thres %f '%confThres
    if forceCPU:
        theCommand+='--device cpu '
    splitOutput=os.path.split(outputPath)
    theCommand+='--project %s --name %s'%(splitOutput[0],splitOutput[1])
    # Execute the command line
    print('[DETECTING] DETECT COMMAND IS %s'%theCommand)
    execute_command(theCommand,hideOutput=False)
    # Return the path where results are stored
    pathLabels=os.path.join(outputPath,'labels')
    print('[DETECTING] DETECTION FINISHED. LABELS SAVED AT %s'%pathLabels)
    return pathLabels

# =============================================================================
# DETECT_WITH_TORCH
# Perform object detection without using the detect.py script.
# Input  : pathYOLOv5 - Local path to YOLOv5. Recommended to use the path
#                       provided by install()
#          weightsFile - Path to the trained model weights.
#          inputPath - Path to the folder containing the images to perform
#                      the detection.
#          imgSize - Images width (512, 640, ...) or None to keep the default.
#          confThreshold - Minimum score to consider an object or None to keep
#                          the default.
#          forceCPU - If True, CPU is used even if GPU is available. If False,
#                     YOLOv5 will select the device (CPU, GPU, ...).
#          labelFormat - The desired format of the provided labels.
#                        0: ABSOLUTE (xyxy), 1: YOLOv5 (xywhn)
#          acceptableExtensions - List of file extensions (without the dot and
#                        uppercase) of the acceptable input images.
# Output : allNames - The base names of each tested file (name without path nor
#                     extension).
#          allTimes - The prediction times of each tested file (ms).
#          allBoxes - Predictions. If labelFormat == 0, the predictions are
#                     an nparray of Nx6, N being the number of detected
#                     objects. For each object, columns contain (in this order)
#                     class, xleft, ytop, xright, ybottom, confidence score.
#                     Coordinates are absolute (pixels).
#                     if labelFormat==1, also Nx6 nparray. Each column is:
#                     class, xcenter, ycenter, width, height, confidence score.
#                     Coordinates and sizes are relative (valued from 0 to 1)
#                     to the image size.
# =============================================================================
def detect_with_torch(pathYOLOv5,weightsFile,inputPath,imgSize=None,confThres=None,forceCPU=False,labelFormat=0,acceptableExtensions=['PNG','JPG']):
    # Get the matplotlib backend (PyTorch changes it or something)
    theBackend=plt.get_backend()
    # Prepare the model and parametrize it
    theModel=torch.hub.load(pathYOLOv5,'custom',source='local',path=weightsFile,force_reload=True)
    #theModel=torch.load(weightsFile)
    if not (confThres is None):
        theModel.conf=confThres
    if forceCPU:
        theModel.cpu()
    # Get all image file names and prepare storage
    allImageFileNames=[os.path.join(inputPath,x) for x in os.listdir(inputPath) if x[-3:].upper() in acceptableExtensions]
    allNames=[]
    allTimes=[]
    allBoxes=[]
    print('[INFO] STARTING INFERENCE ON %d IMAGES'%(len(allImageFileNames)))
    # Loop for all images
    for curImageFileName in allImageFileNames:
        # Get the base name and store it
        baseName=os.path.split(curImageFileName)[1][:-4]
        allNames.append(baseName)
        # Get the prediction
        thePrediction=theModel(imread(curImageFileName))
        # Pick the time
        theTime=sum(thePrediction.t)
        # Pick the bounding boxes in YOLOv5 format
        if labelFormat==0:
            labelData=thePrediction.xyxy[0]
        else:
            labelData=thePrediction.xywhn[0]
        theBoxes=np.roll(labelData.detach().cpu().numpy(),1)
        # Store them
        allTimes.append(theTime)
        allBoxes.append(theBoxes)
    # Restore the matplotlib backend
    matplotlib.use(theBackend)
    print('[INFO] INFERENCE FINISHED.')
    # Return names, times and bounding boxes
    return allNames,allTimes,allBoxes

###############################################################################
# HELPER FUNCTIONS
###############################################################################

# =============================================================================
# EXECUTE_COMMAND
# Wrapper to "simulate" the os.system() command using the safer "subprocess"
# and allowing to hide the output.
# Input  : theCommand - String with the command to issue to the OS.
#          hideOutput - Hide output (True) or not (False)
# =============================================================================
def execute_command(theCommand,hideOutput=True):
    if subprocess.call(theCommand.split(),stdout=[None,subprocess.DEVNULL][hideOutput],stderr=subprocess.STDOUT)!=0:
        sys.exit('[ERROR] UNABLE TO EXECUTE SYSTEM COMMAND %s'%theCommand)

# =============================================================================
# BUILD_DATASET_YAML_FILE
# Builds the YAML file required for YOLOv5 train.py specifying the dataset
# training, validation and test sets. The dataset structure and format is
# assumed to be correct.
# Input  : dataSetPath - Path to the dataset directory containing the 'images'
#                        and labels subfolders.
#          YOLOv5Path - The path where YOLOv5 is installed.
#          classNames - List of strings with the class names.
#          yamlFileName - YAML file name. Recommended to NOT have path. If
#                       None, the file name is constructed using the first
#                       letters of each class followed by "_SPEC.yaml"
# Output : yamlFileName - The used yaml file name.
# =============================================================================
def build_dataset_yaml_file(dataSetPath,YOLOv5Path,classNames,yamlFileName=None):
    relDataSetPath=os.path.relpath(dataSetPath,YOLOv5Path)
    if yamlFileName is None:
        yamlFileName=''.join([x[0] for x in classNames])+'_SPEC.yaml'
    strOut='train: %s\nval: %s\ntest: %s\n\nnc: %d\nnames: %s\n'%(os.path.join(relDataSetPath,'images','train'),os.path.join(relDataSetPath,'images','validation'),os.path.join(relDataSetPath,'images','test'),len(classNames),classNames.__str__())
    with open(yamlFileName,'wt') as outFile:
        outFile.write(strOut)
    return yamlFileName