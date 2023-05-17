#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : odmetrics
# Description : Miscellaneous functions to compute object detection metrics.
# Author      : Antoni Burguera - antoni dot burguera at uib dot es
# History     : 27-Dec-2022 - Creation.
#               17-Feb-2023 - Refactor.
#               16-May-2023 - Corrected error computing numGTLabels in
#                             compute_metrics.
#                             Considered some special cases in compute_map.
#                             Considered empty file in load_yolo_labels.
#                             Added function build_labeled_image.
#                             Minor improvements to plot_rp_curve.
#               17-May-2023 - Added get_best_configuration_multiclass
#                             Updated get_best_configuration to minimize
#                             false positives and negatives.
#                             Updated print_explored_configuration to work
#                             with get_best_configuration_multiclass.
#                             Added group_exploration
#                             Added plot_exploration_multiclass
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from prettytable import PrettyTable
import cv2

###############################################################################
# MAIN FUNCTIONS
###############################################################################

# =============================================================================
# COMPUTE_METRICS
# Computes some quality metrics for object detection
# Input  : predLabels - List of predicted labels. Each item in the list cor-
#                       responds to an image and each label in the item to
#                       a detected object. Labels must be in ABSOLUTE format
#                       with confidence score: [CLASS,XL,YT,XR,YB,CONF.SCORE]
#          gtLabels - List of ground truth labels. Same format as predLabels
#                     except for the confidence score, which should not be
#                     there (it can be, but it does not make sense).
#          iouThresholds - Iterable (list, tuple, ...) containing the IoU
#                     thresholds to evaluate.
#          theClasses - Iterable (list, tuple, ...) containing the classes
#                     to evaluate.
# Output : allResults - List of results. Each item in the list is [c,t,r,p,ap]
#                       where c and t are the class and the IoU threshold
#                       leading to the results. r and p are the list of recalls
#                       and precisions ready to plot the Recall-Precision
#                       curves and ap is the average precision computed by
#                       integrating using the trapezoid rule. Based on this
#                       information, an external program can compute the mAP in
#                       all of its variants (I hope... it seems that every new
#                       object detection work comes with its own definition of
#                       mAP...).
# =============================================================================
def compute_metrics(predLabels,gtLabels,iouThresholds,theClasses):
    numImages=len(predLabels)
    allResults=[]
    for curClass in theClasses:
        forwardData=[]
        for idxImage in range(numImages):
            # To compute true and false positives
            forwardAssociation=associate_labels(predLabels[idxImage],gtLabels[idxImage],curClass)
            # Store confidence score and best IoU
            forwardData+=[[predLabels[idxImage][idxIoU,-1],curIoU] for idxIoU,curIoU in enumerate(forwardAssociation)]
        # Sort it by score
        forwardData=sorted(forwardData, key=lambda theItem: theItem[0], reverse=True)
        # If no associations at all, that's all.
        if len(forwardData)==0:
            for curThreshold in iouThresholds:
                allResults.append([curClass,curThreshold,[0],[0],0])
            break
        # Precompute values
        # Number of ground truth items for the current class
        numGTLabels=len([lblData for imgData in gtLabels for lblData in imgData if lblData[0]==curClass])
        # Number of associated items
        numDataItems=len(forwardData)

        # For each threshold
        for curThreshold in iouThresholds:
            # To integrate, we must ensure the whole Recall-Precision curve is used.
            # To this end, the first recall is set to 0 and the first precision to 1.
            # At the end, a precision of 0 and a recall of 1 is to be added. That
            # is why the lists have numDataItems +2.
            curRecalls=[0]*(numDataItems+2)
            curPrecisions=[1]*(numDataItems+2)
            # Init true positive count
            tpCount=0
            # Loop for all the data
            for iData,theData in enumerate(forwardData):
                tpCount+=(theData[-1]>curThreshold)
                # Store precision and recall
                curPrecisions[iData+1]=tpCount/(iData+1)
                curRecalls[iData+1]=tpCount/numGTLabels
            # Add the final items (precision=0 and recall=1) to ensure the full curve
            # is integrated.
            curPrecisions[numDataItems+1]=0
            curRecalls[numDataItems+1]=1
            # Trapezoidal integral
            curAP=0
            for i in range(1,len(curRecalls)):
                curAP+=(curRecalls[i]-curRecalls[i-1])*(curPrecisions[i]+curPrecisions[i-1])/2
            # Store results
            allResults.append([curClass,curThreshold,curRecalls[1:-1],curPrecisions[1:-1],curAP])
    return allResults

# =============================================================================
# EXPLORE_PARAMETERS
# Computes some basic quality metrics for different combinations of
# IoU and confidence thresholds.
# Input  : predLabels - List of predicted labels. Each item in the list cor-
#                       responds to an image and each label in the item to
#                       a detected object. Labels must be in ABSOLUTE format
#                       with confidence score: [CLASS,XL,YT,XR,YB,CONF.SCORE]
#          gtLabels - List of ground truth labels. Same format as predLabels
#                     except for the confidence score, which should not be
#                     there (it can be, but it does not make sense).
#          iouThresholds - Iterable (list, tuple, ...) containing the IoU
#                     thresholds to evaluate.
#          confThresholds - Iterable (list, tuple, ...) containing the confi-
#                     dence score thresholds to evaluate.
#          theClasses - Iterable (list, tuple, ...) containing the classes
#                     to evaluate.
# Output : List of evaluations where each evaluation is [c,IoU,score,tp,fp,fn,
#          recall, precision,F1-Score], where c, IoU and score are the class,
#          IoU threshold and confidence score threshold leading to this
#          evaluation. tp, fp and fn are the true positives, the false posi-
#          tives and the false negatives respectively. recall and precision
#          are... the recall and the precision. F1-Score is the F1-Score or
#          zero if precision and recall are both zero.
# Note   : This function can be used to find the specific thresholds leading
#          to better results (for example, those maximizing the precision, or
#          the recall, ...)
# =============================================================================
def explore_parameters(predLabels,gtLabels,iouThresholds,confThresholds,theClasses):
    numImages=len(predLabels)
    outData=[]
    # Loop for all classes, IoU thresholds and confidence thresholds
    for curClass in theClasses:
        for curIoU in iouThresholds:
            for curConf in confThresholds:
                # Initialize tp, fp and fn
                tp=fp=fn=0
                # Now loop for each image
                for idxImage in range(numImages):
                    # Get the predicted labels with enough score
                    curPredLabels=[x for x in predLabels[idxImage] if x[-1]>=curConf]
                    # Get the GT labels
                    curGTLabels=gtLabels[idxImage]
                    # Associate each prediction with enough score to zero or
                    # one GT.
                    forwardAssociation=associate_labels(curPredLabels,curGTLabels,curClass)
                    # Associate each GT with zero or one predictions.
                    backwardAssociation=associate_labels(curGTLabels,curPredLabels,curClass)
                    # Associated predictions with enough IoU are true positive.
                    curTP=len([x for x in forwardAssociation if x>=curIoU])
                    # Associated predictions with not enough IoU or not
                    # associated predictions are false positive.
                    curFP=(len(forwardAssociation)-curTP)
                    # GT items associated with a prediction with too low
                    # IoU or not associated at all are false negatives.
                    curFN=len([x for x in backwardAssociation if x<curIoU])
                    # Update counters
                    tp+=curTP
                    fp+=curFP
                    fn+=curFN
                # Compute precision and recall for this configuration
                thePrecision=tp/(tp+fp)
                theRecall=tp/(tp+fn)
                # Safely compute F1-Score
                if thePrecision+theRecall==0:
                    theF1=0
                else:
                    theF1=2*(thePrecision*theRecall)/(thePrecision+theRecall)
                # Store the metrics as well as the corresponding parameter
                # combination.
                outData.append([curClass,curIoU,curConf,tp,fp,fn,theRecall,thePrecision,theF1])
    # That's all. Return the data.
    return outData

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# =============================================================================
# COMPUTE_MAP
# Computes the mAP over all the provided IoU thresholds and classes.
# Input  : theResults - Output of compute_metrics
#          iouThresholds - IoU thresholds for which to compute mAP. These
#                          thresholds must be part (or all) of those used when
#                          calling compute_metrics.
#          theClasses - The classes over which compute the mAP.
# Output : theMap - The requested mAP.
# =============================================================================
def compute_mAP(theResults,iouThresholds,theClasses):
    outData=[]
    for curResult in theResults:
        if curResult[1] in iouThresholds and curResult[0] in theClasses:
            outData.append(curResult[-1])
    if len(outData)==0:
        return 0
    return np.mean(outData)

# =============================================================================
# GET_BEST_CONFIGURATION
# Finds the best configuration for the specified class and target.
# Input  : theExploration - Output of explore_parameters
#          targetMetric - The specific metric to maximize:
#                         0: True Positives
#                         1: False Positives
#                         2: False Negatives
#                         3: Recall
#                         4: Precision
#                         5: F1-Score
#          theClass - The class for which show the results
# Output : The full optimal row of theExploration.
# =============================================================================
def get_best_configuration(theExploration,targetMetric=5,theClass=0):
    # Predefine stuff
    confToMaximize=[0,3,4,5]
    # Maximize or minimize
    if targetMetric in confToMaximize:
        return theExploration[np.argmax([x[targetMetric+3] for x in theExploration if x[0]==theClass])]
    else:
        return theExploration[np.argmin([x[targetMetric+3] for x in theExploration if x[0]==theClass])]

# =============================================================================
# GET_BEST_CONFIGURATION_MULTICLASS
# Finds the best configuration for the specified class/es and target. If more
# than one class is provided, the configuration es searched for all the classes
# together.
# Input  : theExploration - Output of explore_parameters
#          targetMetric - The specific metric to maximize:
#                         0: True Positives
#                         1: False Positives
#                         2: False Negatives
#                         3: Recall
#                         4: Precision
#                         5: F1-Score
#          theClasses - List of class id to explore
# Output : The full optimal row of theExploration except for the first
#          cell, which, instead of containing one integer (the class id) it
#          contains a list (theClasses).
# =============================================================================
def get_best_configuration_multiclass(theExploration,targetMetric,theClasses):
    # Predefine stuff
    confToMaximize=[0,3,4,5]
    # Get TP, FP, FN grouped by classes
    confDict=group_exploration(theExploration,theClasses)
    # Compute precision, recall and F1-Score
    for curKey in confDict:
        [tp,fp,fn]=confDict[curKey]

        thePrecision=tp/(tp+fp)
        theRecall=tp/(tp+fn)
        # Safely compute F1-Score
        if thePrecision+theRecall==0:
            theF1=0
        else:
            theF1=2*(thePrecision*theRecall)/(thePrecision+theRecall)
        confDict[curKey]=[tp,fp,fn]+[thePrecision,theRecall,theF1]
    # Convert to nparray. Each item list is:
    # IoU,ConfThreshold,tp,fp,fn,precision, recall,F1-Score
    theData=np.array([[*k,*v] for k,v in confDict.items()])
    # Maximize or minimize
    if targetMetric in confToMaximize:
        idxOpt=np.argmax(theData[:,2+targetMetric])
    else:
        idxOpt=np.argmin(theData[:,2+targetMetric])

    return [theClasses]+list(theData[idxOpt,:])

###############################################################################
# LABEL MANAGEMENT FUNCTIONS
###############################################################################

# =============================================================================
# CONVERT_LABELS
# Convert the labels format. Possible formats are:
# YOLOV5: CLASS,XCREL,YCREL,WREL,HREL,[SCORE_OPTIONAL]
# ABSOLUTE: CLASS,XLEFT,YTOP,XRIGHT,YBOTTOM,[SCORE_OPTIONAL]
# Input  : theLabels - The labels.
#          imgWidth,imgHeight - Width and height of the labeled image.
#          theConversion - 0: YOLOv5 (xwwhn) to ABSOLUTE (xyxy)
#                          1: ABSOLUTE (xyxy) to YOLOv5 (xwwhn)
# Output : newLabels - Labels in the new format.
# =============================================================================
def convert_labels(theLabels,imgWidth,imgHeight,theConversion):
    if theConversion==0:
        newLabels=[[int(curLabel[0]),
                    int(round((curLabel[1]-curLabel[3]/2)*(imgWidth-1))),
                    int(round((curLabel[2]-curLabel[4]/2)*(imgHeight-1))),
                    int(round((curLabel[1]+curLabel[3]/2)*(imgWidth-1))),
                    int(round((curLabel[2]+curLabel[4]/2)*(imgHeight-1)))]
                   for curLabel in theLabels]
    elif theConversion==1:
        newLabels=[[curLabel[0],
                    ((curLabel[1]+curLabel[3])/2)/(imgWidth-1),
                    ((curLabel[2]+curLabel[4])/2)/(imgHeight-1),
                    (curLabel[3]-curLabel[1])/(imgWidth-1),
                    (curLabel[4]-curLabel[2])/(imgHeight-1)]
                   for curLabel in theLabels]
    else:
        sys.exit('[ERROR] WRONG LABEL CONVERSION SPEC %d. VALID VALUES ARE 0 AND 1'%theConversion)
    if theLabels.shape[0]!=0 and theLabels.shape[1]==6:
        newLabels=np.concatenate((newLabels,theLabels[:,-1:]),axis=1)
    return newLabels

# =============================================================================
# LOAD_YOLO_LABELS
# Loads a file containing YOLOv5 formatted labels.
# Input  : fileName - The name of the file to load.
#          outputFormat - 0: ABSOLUTE, 1: YOLOv5 (see CONVERT_LABELS)
#          imgWidth,imgHeight - Size of the labeled image
# Output : theLabels - Nx5 numpy array, where N is the number of labels
#                      or empty array if no labels.
# =============================================================================
def load_yolo_labels(fileName,outputFormat,imgWidth,imgHeight):
    if os.path.exists(fileName):
        # Load and dismiss warnings if empty file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theLabels=np.loadtxt(fileName,delimiter=' ')
        if len(theLabels)!=0 and len(theLabels.shape)==1:
            theLabels=theLabels.reshape((1,theLabels.shape[0]))
        if outputFormat==0:
            theLabels=convert_labels(theLabels,imgWidth,imgHeight,outputFormat)
    else:
        theLabels=[]
    return np.array(theLabels)

# =============================================================================
# LOAD_ALL_YOLO_LABELS
# Loads all the YOLO label files in a path.
# Input  : gtPath - Path containing the YOLOv5 label files.
#          baseNames - Names of the label files to read (without path or
#          extension.
#          imgWidth,imgHeight - Width and height of the involved images.
#          outputFormat - 0: ABSOLUTE (xyxy), 1: YOLOv5 (xywhn)
#          gtExtension - Extension of the ground truth files (without the dot)
# Output : allGT - list of nparrays. Each list item corresponds to one label
#                  file. Each item is Nx5, N being the number of detected
#                  objects. If outputFormat==0, the columns are: class, xleft
#                  ytop, xright, ybottom (pixel scale). If outputFormat==1
#                  the columns are class, xcenter, ycenter, width, height
#                  in [0,1] scale.
# Note   : This function assumes that all the label files refer to images of
#          the same size. For datasets with images of different sizes, feel
#          free to to it yourself.
# =============================================================================
def load_all_yolo_labels(gtPath,baseNames,imgWidth,imgHeight,outputFormat=0,gtExtension='txt'):
    allGT=[]
    for curName in baseNames:
        gtFileName=os.path.join(gtPath,curName+'.'+gtExtension)
        allGT.append(load_yolo_labels(gtFileName,outputFormat,imgWidth,imgHeight))
    return allGT

###############################################################################
# PLOTTING FUNCTIONS
###############################################################################

# =============================================================================
# PLOT_LABELED_IMAGE
# Plots an image with the specified labels (YOLOv5 format) overlayed.
# Input  : theImage - The image to show.
#          theLabels - List of annotations in ABSOLUTE format (see
#                      CONVERT_LABELS)
#          showClass - Show the class ID (ugly)
#          boxColor - matplotlib color spec to draw the labels boxes.
# =============================================================================
def plot_labeled_image(theImage,theLabels,showClass=False,boxColor='w'):
    plt.figure()
    theAxes=plt.gca()
    if len(theImage.shape)==2 or theImage.shape[2]==1:
        plt.imshow(theImage,cmap='gray')
    else:
        plt.imshow(theImage)
    for curLabel in theLabels:
        curLabel=np.round(curLabel)
        curRect=patches.Rectangle((curLabel[1],curLabel[2]),curLabel[3]-curLabel[1],curLabel[4]-curLabel[2],edgecolor=boxColor,fill=False)
        theAxes.add_patch(curRect)
        if showClass:
            theAxes.annotate('%d'%curLabel[0],(curLabel[1]*(theImage.shape[1]-1),curLabel[2]*(theImage.shape[0]-1)),ha='center',va='center',color=boxColor)
    plt.show()

# =============================================================================
# build_labeled_image
# Plots labels (in ABSOLUTE format) into the image. Modifies the input image.
# Differs from plot_labeled_image in:
# * build_labeled_image does not plot anything, just modifies the image to
#   include the labels.
# * build_labeled_image displays the bounding boxes in a format closer to
#   that of YOLOv5.
# * build_labeled_image uses OpenCV (that's neither good nor bad, just a
#   convenience to keep close to an existing implementation. See Note.)
# Input  : theImage - Image to annotate.
#          theLabels - Labels in ABSOLUTE format.
# Output : No explicit output, but modifies the input image.
# Note   : Based on https://github.com/waittim/draw-YOLO-box
# =============================================================================
def build_labeled_image(theImage,theLabels):
    lineThickness=round(0.002*(theImage.shape[0]+theImage.shape[1])/2)+1
    fontThickness=max(lineThickness-1,1)
    for curLabel in theLabels:
        startPoint=(int(curLabel[1]),int(curLabel[2]))
        endPoint=(int(curLabel[3]),int(curLabel[4]))
        cv2.rectangle(theImage,startPoint,endPoint,color=255,thickness=lineThickness,lineType=cv2.LINE_AA)
        txtLabel=str(int(curLabel[0]))
        txtSize=cv2.getTextSize(txtLabel,0,fontScale=lineThickness/3,thickness=fontThickness)[0]
        endPoint=startPoint[0]+txtSize[0],startPoint[1]-txtSize[1]-3
        cv2.rectangle(theImage,startPoint,endPoint,255,-1,cv2.LINE_AA)
        cv2.putText(theImage,txtLabel,(startPoint[0],startPoint[1]-2),0,lineThickness/3,0,thickness=fontThickness,lineType=cv2.LINE_AA)

# =============================================================================
# PLOT_RP_CURVES
# Plot the Recall-Precision curves for the requested class and IoU values.
# Input  : theResults - Output of compute_metrics.
#          iouValues - Iterable of IoU values to use.
#          theClass - Iterable of classes to use. If single class, an int is
#                     also allowed.
#          theTitle - Figure title or None to not have title.
# =============================================================================
def plot_rp_curves(theResults,iouValues,theClass=0,theTitle=None):
    # Wrapper to allow programs using previous versions to work.
    if type(theClass) is int:
        theClass=[theClass]
    # Get the curves
    outCurves=[]
    outLegends=[]
    for curIoU in iouValues:
        curResult=next(([x[2],x[3],x[4]] for x in theResults if x[0] in theClass and x[1]==curIoU),None)
        outCurves.append(curResult)
        outLegends.append('IoU>=%.3f, AP=%.3f'%(curIoU,curResult[-1]))
    # Check possible error
    if None in outCurves:
        sys.exit('[ERROR] ATTEMPT TO PLOT AN RECALL-PRECISION CURVE FOR A NON-RECORDED IOU LEVEL.')
    # Plot the curves
    plt.figure()
    for curCurve in outCurves:
        plt.plot(curCurve[0],curCurve[1])
    plt.legend(outLegends)
    plt.grid('on')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if not (theTitle is None):
        plt.title(theTitle)
    plt.show()

# =============================================================================
# PLOT_EXPLORATION
# Plots the results of explore_parameters.
# Input  : theExploration - Output of explore_parameters
#          metricToShow - The specific metric to show:
#                         0: True Positives
#                         1: False Positives
#                         2: False Negatives
#                         3: Recall
#                         4: Precision
#                         5: F1-Score
#          theClass - The class for which show the results
# =============================================================================
def plot_exploration(theExploration,metricToShow=5,theClass=0):
    # Define metric names
    metricNames=['TP','FP','FN','RECALL','PRECISION','F1-SCORE']
    # Get X and Y values
    theX=np.sort(np.unique([x[1] for x in theExploration if x[0]==theClass]))
    theY=np.sort(np.unique([x[2] for x in theExploration if x[0]==theClass]))
    # Build the Z array matching the values
    theZ=np.array([[next(x[metricToShow+3] for x in theExploration if x[0]==theClass and x[1]==curX and x[2]==curY) for curX in theX] for curY in theY])
    # Meshgrid the X and the Y
    theX,theY=np.meshgrid(theX,theY)
    # Create the figure and plot everything.
    theFigure=plt.figure()
    theAxes=theFigure.add_subplot(projection='3d')
    theAxes.plot_surface(theX,theY,theZ)
    theAxes.set_xlabel('IOU THRESHOLD')
    theAxes.set_ylabel('SCORE THRESHOLD')
    theAxes.set_zlabel(metricNames[metricToShow])
    plt.show()

# =============================================================================
# PLOT_EXPLORATION_MULTICLASS
# Plots the results of explore_parameters grouping data in the specified
# classes-
# Input  : theExploration - Output of explore_parameters
#          metricToShow - The specific metric to show:
#                         0: True Positives
#                         1: False Positives
#                         2: False Negatives
#                         3: Recall
#                         4: Precision
#                         5: F1-Score
#          theClasses - List of classes to consider.
# =============================================================================
def plot_exploration_multiclass(theExploration,metricToShow,theClasses):
    # Define metric names
    metricNames=['TP','FP','FN','RECALL','PRECISION','F1-SCORE']
    # Get the exploration data grouped by classes
    confDict=group_exploration(theExploration,theClasses)
    # Compute the desired metric
    for curKey in confDict:
        [tp,fp,fn]=confDict[curKey]
        if metricToShow==0:
            confDict[curKey]=tp
        elif metricToShow==1:
            confDict[curKey]=fp
        elif metricToShow==2:
            confDict[curKey]=fn
        elif metricToShow==3:
            confDict[curKey]=tp/(tp+fn)
        elif metricToShow==4:
            confDict[curKey]=tp/(tp+fp)
        elif metricToShow==5:
            thePrecision=tp/(tp+fp)
            theRecall=tp/(tp+fn)
            if thePrecision+theRecall==0:
                theF1=0
            else:
                theF1=2*(thePrecision*theRecall)/(thePrecision+theRecall)
            confDict[curKey]=theF1
    # Convert to nparray
    theData=np.array([[*k,v] for k,v in confDict.items()])

    # Get X and Y values
    theX=np.sort(np.unique([x[0] for x in theData]))
    theY=np.sort(np.unique([x[1] for x in theData]))
    # Build the Z array matching the values
    theZ=np.array([[next(x[2] for x in theData if x[0]==curX and x[1]==curY) for curX in theX] for curY in theY])
    # Meshgrid the X and the Y
    theX,theY=np.meshgrid(theX,theY)
    # Create the figure and plot everything.
    theFigure=plt.figure()
    theAxes=theFigure.add_subplot(projection='3d')
    theAxes.plot_surface(theX,theY,theZ)
    theAxes.set_xlabel('IOU THRESHOLD')
    theAxes.set_ylabel('SCORE THRESHOLD')
    theAxes.set_zlabel(metricNames[metricToShow])
    plt.show()

# =============================================================================
# PRINT_EXPLORED_CONFIGURATION
# Simple helper function to print one row of the explore_parameters output.
# Input  : exploredConfiguration - Row of explore_parameters.
# =============================================================================
def print_explored_configuration(exploredConfiguration):
    theConfiguration=exploredConfiguration.copy()
    if type(theConfiguration[0]) is int:
        theConfiguration[0]=str(exploredConfiguration[0])
    else:
        theConfiguration[0]=','.join([str(x) for x in exploredConfiguration[0]])
    strHeaders=['* CLASS           : %s',
                '* IOU THRESHOLD   : %.3f',
                '* SCORE THRESHOLD : %.3f',
                '* TP              : %d',
                '* FP              : %d',
                '* FN              : %d',
                '* RECALL          : %.3f',
                '* PRECISION       : %.3f',
                '* F1-SCORE        : %.3f'
                ]
    for i,v in enumerate(theConfiguration):
        print(strHeaders[i]%v)

# =============================================================================
# PRINT_EXPLORATION_SUMMARY
# For the specified metric,  searches the optimal configuration for each class
# (individually) and all together. Prints a table summarizing the results.
# Input  : theExploration - Output of explore_parameters
#          classesToEvaluate - List of class id to evaluate.
#          classNames - Names (to print) of the classes in classesToEvaluate
#          metricToOptimize - The metric to optimize:
#                               0: True Positives
#                               1: False Positives
#                               2: False Negatives
#                               3: Recall
#                               4: Precision
#                               5: F1-Score
# =============================================================================
def print_exploration_summary(theExploration,classesToEvaluate,classNames,metricToOptimize=5):
    theTable=PrettyTable(['CLASS','IoU','CONF','TP','FP','FN','R','P','F1-SCORE'])
    for iClass,curClass in enumerate(classesToEvaluate):
        bestConfig=get_best_configuration_multiclass(theExploration,metricToOptimize,[curClass])
        theTable.add_row([classNames[iClass],round(bestConfig[1],3),round(bestConfig[2],3),int(bestConfig[3]),int(bestConfig[4]),int(bestConfig[5]),round(bestConfig[6],3),round(bestConfig[7],3),round(bestConfig[8],3)])
    bestConfig=get_best_configuration_multiclass(theExploration,metricToOptimize,classesToEvaluate)
    theTable.add_row(['ALL',round(bestConfig[1],3),round(bestConfig[2],3),int(bestConfig[3]),int(bestConfig[4]),int(bestConfig[5]),round(bestConfig[6],3),round(bestConfig[7],3),round(bestConfig[8],3)])
    print(theTable)

###############################################################################
# AUXILIARY FUNCTIONS
###############################################################################

# =============================================================================
# COMPUTE_IOU
# Computes the Intersection over Union (IoU) between two labels. The labels
# must be ABSOLUTE (see CONVERT_LABELS), meaning that they have XL,YT,XR,YB
# at indices 1,2,3 and 4.
# Input  : firstLabel - One (ABSOLUTE) label.
#          secondLabel - The other (ABSOLUTE) label.
# Output : theIOU - The Intersection over Union
# =============================================================================
def compute_IoU(firstLabel,secondLabel):
    # Compute coordinates of the intersection rectangle
    Xl=max(firstLabel[1],secondLabel[1])
    Yt=max(firstLabel[2],secondLabel[2])
    Xr=min(firstLabel[3],secondLabel[3])
    Yb=min(firstLabel[4],secondLabel[4])
    # Compute the intersection area
    theIntersection=max(0,(Xr-Xl)+1)*max(0,(Yb-Yt)+1)
    # If the intersection is 0, nothing else to do.
    if theIntersection==0:
        return 0
    # Compute the boxes area
    firstArea=(firstLabel[3]-firstLabel[1]+1)*(firstLabel[4]-firstLabel[2]+1)
    secondArea=(secondLabel[3]-secondLabel[1]+1)*(secondLabel[4]-secondLabel[2]+1)
    # The area of the union is the sum of the areas minus that of the
    # intersection.
    theUnion=firstArea+secondArea-theIntersection
    # Return the IoU
    return theIntersection/theUnion

# =============================================================================
# ASSOCIATE_LABELS
# Given two sets of labels, computes the maximum IoU of each label in the
# first set and all the labels in the second set.
# Input  : lblA- First set (ABSOLUTE format)
#          lblB - Second set (ABSOLUTE format)
#          theClass - Class to consider
# Output : allIoU - For each label in lblA, it contains the largest IoU with
#                   the labels in lblB
# =============================================================================
def associate_labels(lblA,lblB,theClass=0):
    return [max([compute_IoU(x,y) for y in lblB if y[0]==theClass],default=0) for x in lblA if x[0]==theClass]

# =============================================================================
# GROUP_EXPLORATION
# Given the results of explore_parameters, groups the results (only tp, fp and
# fn) by the specified class or classes.
# Input : theExploration - Output of explore_parameters
#       : theClasses - List with the class or classes to group the data.
# =============================================================================
def group_exploration(theExploration,theClasses):
    # Get TP, FP, FN grouped by classes
    confDict={}
    for curData in theExploration:
        if curData[0] in theClasses:
            theKey=tuple(curData[1:3])
            if theKey in confDict:
                [tp,fp,fn]=confDict[theKey]
            else:
                tp=fp=fn=0
            tp+=curData[3]
            fp+=curData[4]
            fn+=curData[5]
            confDict[theKey]=[tp,fp,fn]
    return confDict