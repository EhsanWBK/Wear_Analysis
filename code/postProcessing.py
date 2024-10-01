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
from os import makedirs, listdir
from os.path import join, dirname
import csv

def measurementVB(frame: ndarray) -> list:
    ''' Take in predicted mask. Find contours and draw box around them to highlight region of interest (roi).
    Align mask pixels to box edge and calcuate distances to the edge.
    Create list of all distances. Accumulated these distances create the Width of Flank Wear Land VB.

    Returns:
    - VB_max: biggest VB value
    - VB_B: average VB value (not implemented yet)
    - ROI image for display (not implemented yet)'''

    maxVBList = []
    maxVB = []
    aspectRatio = (2000,1400)
    count = 0
    img = cv2.resize(frame, aspectRatio, interpolation=cv2.INTER_LINEAR)
    
    VBmax = []
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = getCountourIndices(contours)
    #print(result)
    min_max_list = []
    VBmax_AX = []

    for i in result:
        min_max_list = []
        CNT = contours[i]
        M = cv2.moments(CNT)
        #print(M)
        RECT = cv2.minAreaRect(CNT)
        BOX = cv2.boxPoints(RECT)
        BOX = np.int0(BOX)
        FIT = cv2.drawContours(np.copy(img), [BOX], 0, (255, 255, 255), 1)
        # output_path_fit = os.path.join(results_folder, f"{filename[:-4]}_{i}_fit.png")
        # cv2.imwrite(output_path_fit, FIT)

        coordinates = BOX.reshape((-1, 1, 2))

        # Create a mask image with the same dimensions as the original image
        mask = np.zeros_like(img)

        # Draw a filled polygon (ROI) on the mask using the defined coordinates
        mask = cv2.fillPoly(mask, [coordinates], (255, 255, 255))

        # Perform a bitwise AND operation between the original image and the mask to extract the ROI
        roi = cv2.bitwise_and(img, img, mask=mask)

        edges = cv2.Canny(roi, 180, 500)
        # output_path_canny = os.path.join(results_folder, f"{filename[:-4]}_{i}_fit_canny.png")
        # cv2.imwrite(output_path_canny, edges)
        White_Pixels = np.argwhere(edges == 255)
        White_Pixels_tolist = White_Pixels.tolist()
        #print(White_Pixels_tolist)

        for y, x in White_Pixels_tolist:
            found = False
            for entry in min_max_list:
                if entry[0] == y:
                    # Update min and max x values if necessary
                    entry[1] = min(entry[1], x)  # min_x
                    entry[2] = max(entry[2], x)  # max_x
                    found = True
                    break

            if not found:
                # If y is not already in the list, append a new entry
                min_max_list.append([y, x, x, 0])  # [y, min_x, max_x, difference]

        # Calculate the difference and update the min_max_list
        for entry in min_max_list:
            entry[3] = round(np.cross(BOX[1] - BOX[0], np.array([entry[2], entry[0]]) - BOX[0]) / np.linalg.norm(BOX[1] - BOX[0]), 2)

        # Print the result
        for entry in min_max_list:
            y, min_x, max_x, difference = entry

        Dis = [sublist[3] for sublist in min_max_list]
        VBmax_AX.append(round(float(max(Dis)) * 1.725, 2))
    VBmax.append(max(VBmax_AX))
    print(VBmax)


def getCountourIndices(contours):
    ''' '''
    valid_indices = []
    for i in range(len(contours)):
        if len(contours[i]) > 100: # filter out too small / unimportant contours
            for j in range(len(contours[i])):
                if 1100 < contours[i][j][0][0] + contours[i][j][0][1] < 2000:
                    valid_indices.append(i)
                    break  # Exit the inner loop once condition is met for this contour
    return valid_indices

def wearDetectionStack(dataStack, nrEdges, resultFolder):
    ''' Takes in array of masks. '''
    resultsVBMax = []
    edgeTemp = []
    # for i in range(len(dataStack)//nrEdges):
    #     for edge in range(nrEdges):
    #         edgeTemp=[]
    #         sampleVBMax = measurementVB(frame=dataStack[(i-1)*nrEdges+edge])
    #         edgeTemp.append(sampleVBMax)
    #     resultsVBMax.append(edgeTemp)
    for idx in range(len(dataStack)):
        sampleVBMax = measurementVB(frame=dataStack[idx])
        resultsVBMax.append(sampleVBMax)
    resultFolder = writeCSV(resultsVBMax=resultsVBMax, resultFolder=resultFolder)
    return resultsVBMax, resultFolder

def writeCSV(resultsVBMax, resultFolder: str):
    ''' '''
    # Define the header
    header = ['Tooth'] + [f"{i+1}_Track" for i in range(len(resultsVBMax)//4)]

    # Initialize the rows with tooth labels and data
    rows = [
        ['1st Cutting Edge'] + [float(resultsVBMax[i*4]) for i in range(len(resultsVBMax)//4)],
        ['2nd Cutting Edge'] + [float(resultsVBMax[(i*4)+1]) for i in range(len(resultsVBMax)//4)],
        ['3rd Cutting Edge'] + [float(resultsVBMax[(i*4)+2]) for i in range(len(resultsVBMax)//4)],
        ['4th Cutting Edge'] + [float(resultsVBMax[(i*4)+3]) for i in range(len(resultsVBMax)//4)],
    ]

    # Define the path to save the CSV file
    csvFolder = join(resultFolder, 'wearCurve'+str(getTimeStamp()))
    makedirs(csvFolder, exist_ok=True)

    # Write data to the CSV file
    with open(csvFolder, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(rows)  # Write the rows
    return csvFolder

def plotWearCurve(filePath):
    df = pd.read_csv(filePath)

    # Creating x-axis labels for all 5 cuts
    x_labels = [f'{i}' for i in range(1, (df.shape[1]-1)*5, 5)]

    # Create the results folder if it doesn't exist
    results_folder = join(dirname(filePath), 'wearCurve'+str(getTimeStamp()))
    makedirs(results_folder, exist_ok=True)

    # Define a list of colors
    colors = ['blue', 'orange', 'green', 'red']

    # Determine the common y-axis limit
    y_min = 0
    y_max = 270

    # Plot the data with markers only
    plt.figure(figsize=(12, 8))
    for index, row in df.iterrows():
        plt.plot(df.columns[1:], row[1:], marker='o', linestyle='None', label=row['Tooth'], color=colors[index % len(colors)])

    # Reduce the number of x-axis labels by selecting a subset
    plt.xticks(ticks=range(0, len(x_labels), 10), labels=[x_labels[i] for i in range(0, len(x_labels), 10)], rotation=45)

    plt.xlabel('Cuts', fontsize=28)
    plt.ylabel('VB (μm)', fontsize=28)
    plt.title('Wear Curve (V$_{f}$=510 mm/min)', fontsize=34)  # Increase the title font size
    plt.legend(loc='upper left',fontsize=24)  # Increase the legend font size
    plt.ylim(y_min, y_max)  # Set common y-axis limit
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True, linestyle='--')  # Add grid lines
    plt.tight_layout()

    # Full path to save the file
    output_path = join(results_folder, 'VB_Curve.svg')

    # Save the plot as a PNG file
    plt.savefig(output_path, format='svg', dpi=300)

    # Show the plot
    plt.show()

    # Plot each tooth separately and save
    for index, row in df.iterrows():
        plt.figure(figsize=(12, 8))
        plt.plot(df.columns[1:], row[1:], marker='o', linestyle='None', label=row['Tooth'], color=colors[index % len(colors)])

        plt.xticks(ticks=range(0, len(x_labels), 10), labels=[x_labels[i] for i in range(0, len(x_labels), 10)], rotation=45)
        
        plt.xlabel('Cuts', fontsize=14)
        plt.ylabel('VB (μm)', fontsize=14)
        plt.title(f'Wear Curve ({row["Tooth"]})', fontsize=16)  # Increase the title font size
        plt.legend(loc='upper left',fontsize=14)  # Increase the legend font size
        plt.ylim(y_min, y_max)  # Set common y-axis limit
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, linestyle='--')  # Add grid lines
        plt.tight_layout()

        # Full path to save the file
        individual_output_path = join(results_folder, f'VB_Curve_Tooth_{row["Tooth"]}.svg')

        # Save the plot as a PNG file
        plt.savefig(individual_output_path, format='svg', dpi=300)
        
        # Show the plot
        plt.show()

# directory = r"C:\Users\flohg\Desktop\Hiwi_WBK\wear_test\opcVersion\data_training\milling\newDataSet\Test1_TOT\Tst"
# print(directory)
# for filename in listdir(directory):
#     if filename.endswith("prediction.png") or filename.endswith("prediction.jpg"):
#         # img_name.append(filename)
#         img = cv2.imread(join(directory, filename), cv2.IMREAD_GRAYSCALE)
#         measurementVB(img)