''' ## Post-Processing of Segmentation

NOT WORKING AT THE MOMENT!

Evaluate segmentation and create wear boundaries.

Accessible Functions:
- 
'''
from generalUtensils import getTimeStamp

import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
import cv2
import pandas as pd
from os import makedirs
from os.path import join
import csv
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess


def measurementVB(frame: ndarray, saveFolder:str, filename:str = 'test') -> list:
    ''' Take in predicted mask. Find contours and draw box around them to highlight region of interest (roi).
    Align mask pixels to box edge and calcuate distances to the edge.
    Create list of all distances. Accumulated these distances create the Width of Flank Wear Land VB.

    Returns:
    - VB_max: biggest VB value
    - VB_B: average VB value (not implemented yet)
    - ROI image for display (not implemented yet)'''

    aspectRatio = (2000,1400)
    img = cv2.resize(frame, aspectRatio, interpolation=cv2.INTER_LINEAR)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = getCountourIndices(contours)
    VBmax_AX = []
    fitImg = []

    for i in result:
        min_max_list = []
        CNT = contours[i]
        RECT = cv2.minAreaRect(CNT)
        BOX = cv2.boxPoints(RECT)
        BOX = np.int0(BOX)
        imgColor = cv2.cvtColor(np.copy(img), cv2.COLOR_GRAY2BGR)
        FIT = cv2.drawContours(np.copy(imgColor), [BOX], 0, (0, 0, 255), 3)
        fitImg.append(FIT)
        

        coordinates = BOX.reshape((-1, 1, 2))
        mask = np.zeros_like(img) # Create a mask image with the same dimensions as the original image
        mask = cv2.fillPoly(mask, [coordinates], (255, 255, 255)) # Draw a filled polygon (ROI) on the mask using the defined coordinates
        roi = cv2.bitwise_and(img, img, mask=mask) # Perform a bitwise AND operation between the original image and the mask to extract the ROI

        edges = cv2.Canny(roi, 180, 500)
        kernel = np.ones((3, 3), np.uint8) # Dilate the edges to increase thickness
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        White_Pixels = np.argwhere(edges_dilated == 255)
        White_Pixels_tolist = White_Pixels.tolist()

        for y, x in White_Pixels_tolist:
            found = False
            for entry in min_max_list:
                if entry[0] == y:
                    entry[1] = min(entry[1], x)  # min_x
                    entry[2] = max(entry[2], x)  # max_x
                    found = True
                    break
            if not found:
                min_max_list.append([y, x, x, 0])  # [y, min_x, max_x, difference]
        for entry in min_max_list:
            entry[3] = round(np.cross(BOX[1] - BOX[0], np.array([entry[2], entry[0]]) - BOX[0]) / np.linalg.norm(BOX[1] - BOX[0]), 2)
        for entry in min_max_list:
            y, min_x, max_x, difference = entry
        Dis = [sublist[3] for sublist in min_max_list]
        VBmax_AX.append(round(float(max(Dis)) * 1.725, 2))
    output_path_fit = join(saveFolder, 'fit' ,'fit_'+filename+".png")
    cv2.imwrite(output_path_fit, fitImg[VBmax_AX.index(max(VBmax_AX))])
    print('VB max: ',VBmax_AX)
    return max(VBmax_AX)

def getCountourIndices(contours):
    valid_indices = []
    for i in range(len(contours)):
        if len(contours[i]) > 100: # filter out too small / unimportant contours
            for j in range(len(contours[i])):
                if 1100 < contours[i][j][0][0] + contours[i][j][0][1] < 2000:
                    valid_indices.append(i)
                    break  # Exit the inner loop once condition is met for this contour
    return valid_indices


def writeCSV(resultsVBMax, resultFolder):
    header = ['Tooth'] + [f"{i+1}_Track" for i in range(len(resultsVBMax)//4)]
    rows = [
        ['1st Cutting Edge'] + [float(resultsVBMax[i*4]) for i in range(len(resultsVBMax)//4)],
        ['2nd Cutting Edge'] + [float(resultsVBMax[(i*4)+1]) for i in range(len(resultsVBMax)//4)],
        ['3rd Cutting Edge'] + [float(resultsVBMax[(i*4)+2]) for i in range(len(resultsVBMax)//4)],
        ['4th Cutting Edge'] + [float(resultsVBMax[(i*4)+3]) for i in range(len(resultsVBMax)//4)],
    ]
    csvFile = join(resultFolder,str(getTimeStamp())+'.csv')
    with open(csvFile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(rows)  # Write the rows
    return resultFolder, csvFile

def plotWearCurve(filePath, resultFolder):
    ''' Plot the Wear Curve. '''
    df = pd.read_csv(filePath)
    colors = ['blue', 'orange', 'green', 'red']
    y_min = 0
    y_max = 400
    xLabel = [f'{i}' for i in range(1, (df.shape[1] - 1) * 5, 5)]

    wearCurveFolder = join(resultFolder, 'wearCurve')
    makedirs(wearCurveFolder, exist_ok=True)

    plt.figure(figsize=(12, 8))
    for index, row in df.iterrows():
        plt.plot(df.columns[1:], row[1:], marker='o', linestyle='None', label=row['Tooth'], color=colors[index % len(colors)])
    
    # Plot Total Wear Curve
    plt.xticks(ticks=range(0, len(xLabel), 10), labels=[xLabel[i] for i in range(0, len(xLabel), 10)], rotation=45)
    plt.xlabel('Cuts', fontsize=28)
    plt.ylabel('VB (μm)', fontsize=28)
    plt.title('Wear Curve (V$_{f}$=510 mm/min)', fontsize=34)  # Increase the title font size
    plt.legend(loc='upper left',fontsize=24)  # Increase the legend font size
    plt.ylim(y_min, y_max)  # Set common y-axis limit
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True, linestyle='--')  # Add grid lines
    plt.tight_layout()
    output_path = join(wearCurveFolder, 'VB_Curve_'+str(getTimeStamp())+'.svg') # Full path to save the file
    plt.savefig(output_path, format='svg', dpi=300) # Save the plot as a SVG file
    # wearCurve = plt.gcf()

    # Plot each tooth separately and save
    for index, row in df.iterrows():
        plt.figure(figsize=(12, 8))
        plt.plot(df.columns[1:], row[1:], marker='o', linestyle='None', label=row['Tooth'], color=colors[index % len(colors)])
        plt.xticks(ticks=range(0, len(xLabel), 10), labels=[xLabel[i] for i in range(0, len(xLabel), 10)], rotation=45)
        plt.xlabel('Cuts', fontsize=14)
        plt.ylabel('VB (μm)', fontsize=14)
        plt.title(f'Wear Curve ({row["Tooth"]})', fontsize=16)  # Increase the title font size
        plt.legend(loc='upper left',fontsize=14)  # Increase the legend font size
        plt.ylim(y_min, y_max)  # Set common y-axis limit
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, linestyle='--')  # Add grid lines
        plt.tight_layout()
        individual_output_path = join(wearCurveFolder, f'VB_Curve_Tooth_{row["Tooth"]}.svg')
        plt.savefig(individual_output_path, format='svg', dpi=300)
    # return wearCurve
    return None

def outlierDetection(filePath, resultFolder, cuttingEdges = 4):
    ''' Clean up Wear Curve from outliers with DBSCAN. '''
    df = pd.read_csv(filePath)
    colors = ['blue', 'orange', 'green', 'red']
    y_min = 0
    y_max = 400
    xLabel = [f'{i}' for i in range(1, (df.shape[1] - 1) * 5, 5)]

    outlierFolder = join(resultFolder, 'outlier')
    makedirs(outlierFolder)
    for idx in range(cuttingEdges):
        wearVal = df.iloc[idx, 1:].to_numpy()
        xVal = np.arange(1, len(wearVal)+1)
        data2D = np.column_stack((xVal, wearVal))
        clustering = DBSCAN(eps=15, min_samples=10).fit(data2D)
        labels = clustering.labels_
        plt.figure(figsize=(12,8))
        classMemberMask = (labels != -1) # cluster
        xy = data2D[classMemberMask]
        plt.scatter(xy[:, 0], xy[:, 1], c=colors[idx], edgecolor='k', label=f'{idx + 1}st Tooth Clusters')
        classMemberMask = (labels == -1) # outlier
        xy = data2D[classMemberMask]
        plt.scatter(xy[:, 0], xy[:, 1], c='red',  marker='X', edgecolor='k', label=f'{idx + 1}st Tooth Outliers')
        plt.xticks(ticks=range(0, len(xLabel), 10), labels=[xLabel[i] for i in range(0, len(xLabel), 10)], rotation=45)
        plt.title(f'VB Curve_DBSCAN Clustering for Tooth {idx + 1}', fontsize=16)        
        plt.xlabel('Cuts', fontsize=14)
        plt.ylabel('VB (μm)', fontsize=14)
        plt.legend(loc='upper left',fontsize=14)
        plt.ylim(y_min, y_max)  # Set common y-axis limit
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        plt.savefig(join(outlierFolder, f'VB_Curve_DBSCAN_Tooth_{idx + 1}.svg'), format='svg', dpi=300)
    return None

def plotWearCurveLOWESS(filePath, resultFolder):
    ''' Plot the Wear Curve with LOWESS Smoothing'''
    df = pd.read_csv(filePath)
    colors = ['blue', 'orange', 'green', 'red']
    y_min = 0
    y_max = 400
    xVal = range(1, df.shape[1])
    xLabel = [f'{i}' for i in range(1, (df.shape[1]-1)*5, 5)]
    wearCurveLowess = join(resultFolder, 'wearCurveLOWESS')
    makedirs(wearCurveLowess)
    plt.figure(figsize=(12, 8))
    smoothingFrac = 0.2
    for index, row in df.iterrows():
        loess_smoothed = lowess(row[1:], xVal, frac=smoothingFrac)
        plt.scatter(xVal, row[1:], marker='o', label=row['Tooth'], color=colors[index % len(colors)])
        plt.plot(loess_smoothed[:, 0], loess_smoothed[:, 1], color=colors[index % len(colors)])
    
    # Plot Total Wear Curve with LOWESS smoothing
    plt.xticks(ticks=range(0, len(xLabel), 10), labels=[xLabel[i] for i in range(0, len(xLabel), 10)], rotation=45)
    plt.xlabel('Cuts', fontsize=14)
    plt.ylabel('VB (μm)', fontsize=14)
    plt.title('VB Curve (LOWESS)', fontsize=16)
    plt.legend(loc='upper left', fontsize=20)
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    output_path = join(wearCurveLowess, 'VB_Curve_LOWESS.svg')
    plt.savefig(output_path, format='svg', dpi=300)

    # Plot each tooth separately and save with LOWESS smoothing
    for index, row in df.iterrows():
        plt.figure(figsize=(12, 8))
        loess_smoothed = lowess(row[1:], xVal, frac=smoothingFrac)
        plt.scatter(xVal, row[1:], marker='o', label=row['Tooth'], color=colors[index % len(colors)])
        plt.plot(loess_smoothed[:, 0], loess_smoothed[:, 1], color=colors[index % len(colors)])
        plt.xticks(ticks=range(0, len(xLabel), 10), labels=[xLabel[i] for i in range(0, len(xLabel), 10)], rotation=45)
        plt.xlabel('Cuts', fontsize=14)
        plt.ylabel('VB (μm)', fontsize=14)
        plt.title(f'VB Curve for {row["Tooth"]} (LOWESS)', fontsize=16)
        plt.legend(loc='upper left', fontsize=14)
        plt.ylim(y_min, y_max)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        individual_output_path = join(wearCurveLowess, f'VB_Curve_Tooth_{row["Tooth"]}_LOWESS.svg')
        plt.savefig(individual_output_path, format='svg', dpi=300)
    return None

