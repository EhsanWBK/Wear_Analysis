"""
Created on Thu Mar 16 11:14:58 2023

@author: programmieren

Updated by: Florian Schindler
"""

# general libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# machine learning libraries
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# local files
import generalUtensils as gu

# preprocessing segmented pixels
class WearClusterer:
    
    def __init__(self, mask) -> None:
        '''
        Create wear cluster based on mask created by pretrained mode
        Goal is to determine wear area
        '''
        global curMask

        self.COLOURMAP_NAME = 'Spectral'    # constant for pyplot colour map 

        self.colorMaskShape = (mask.shape[0], mask.shape[1], 4)

        # coordinates of wear; gathered from masked created in segementation; CoS starts in upper left corner
        maskCOS = np.nonzero(mask)[:2]
        print('Wear Coordinates:\n', maskCOS)

        self.wear_y, self.wear_x = maskCOS  # use x and y coordinates
        self.wearCOS = np.dstack((self.wear_x, self.wear_y))[0] # transform to points (tuple); dimensions reduced
        curMask = mask
        

    def wearClustering(self):
        # cluster wear points
        clusterAlgorithm = DBSCAN(eps=4, min_samples=4) # initialize clustering algorithm
        cluster = clusterAlgorithm.fit(self.wearCOS)    # apply algorithm to coordinates

        # explore clustering (unique labels, total number of labels)
        self.labelList = cluster.labels_    # labels of cluster
        self.uniqueLabels, self.labelCounts = np.unique(self.labelList, return_counts=True)

    def visualizeCluster(self):
        '''
        Visulaization of wear cluster by label with pyplot
        '''

        # define colours in RGBA
        colors = plt.get_cmap(self.COLOURMAP_NAME)(np.linspace(0, 1, len(self.uniqueLabels)))
                
        # write clustering labels in mask: 0: Background; 1: noise/outliers; 2... cluster
        clustered_mask = np.zeros(self.colorMaskShape)

        # add offset of 1 to labels   
        print("\nLabel list: shape ", self.labelList.shape)
        labels_up = self.labelList + np.ones(len(self.labelList))
        print("Labels with Offset: shape ", labels_up.shape)
        
        # take corresponding color for each label
        labels_colored = colors[self.labelList.astype(int)]
        print('Coloured labels: shape ',labels_colored.shape)
        
        clustered_mask[self.wear_y,self.wear_x] = labels_colored
        print("Clustered mask: shape ", clustered_mask.shape) # uncomment for debug
        
        #Color values are normalized
        mask_display = clustered_mask*255

        return(mask_display.astype(int))

    
def get_edges(img: np.ndarray) -> np.ndarray:
    """
    Get edges between background and tool.

    Parameters
    ----------
    img : np.ndarray
        Image.

    Returns
    -------
    kos: np.ndarray:
        List of the coordinates (y, x) of the edges.

    """

    # grayscale image
    imgGray = gu.adapt_channels(img, desired_channels=1)    # reduces channels to 1 (grayscaled image)
    # blur image for noise reduction
    imgBlur = cv2.GaussianBlur(imgGray, ksize=(3,1), sigmaX=0) # ksize: pixel window in which image is blurred
    
    imgEdited = imgBlur
    # edge detection with Canny- or Sobel-Filter
    edges = cv2.Canny(imgEdited, threshold1=240, threshold2=255) # toggle thresholds to adjust to picture quality
    # edges = cv2.Sobel(src=imgEdited, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    
    # gives out wear image with Canny-Filter in the end
    wear = cv2.Canny(imgEdited, threshold1=100, threshold2=200)

    # edge coordinates
    edgeXY = np.nonzero(edges)  # initialzie edge coordinates
    edgeCO = np.dstack((edgeXY[1],edgeXY[0]))[0]
    
    # noise removal
    edgeAlgorithm = DBSCAN(eps=6) # initialize clustering algorithm
    edgeCluster = edgeAlgorithm.fit(edgeCO)

    # explore labels
    labelList = edgeCluster.labels_
    uniqueLabels, labelCounts = np.unique(labelList, return_counts=True)
    maxOccLblIdx = np.argmax(labelCounts)   # index of most occuring labels
    lblCntCpy = labelCounts.copy()
    sndOccLblIdx = np.argmax(np.delete(lblCntCpy, maxOccLblIdx, 0))
    print('\nTool edge clustering (unique labels)\t', uniqueLabels) # uncomment for debug
    print('Tool edge clustering (label counts):\t', labelCounts)
    print('Tool edge clustering (max index):\t', maxOccLblIdx)
    print('Tool edge clustering (2nd index):    \t', sndOccLblIdx)

    # return biggest cluster of coordinates; therefore, biggest edge
    biggestClusterIdx = labelList == (uniqueLabels[maxOccLblIdx]*np.ones(len(labelList), dtype=np.uint8))
    secondBiggestClusterIdx = labelList == (uniqueLabels[sndOccLblIdx]*np.ones(len(labelList), dtype=np.uint8))
    
    return edgeCO[biggestClusterIdx], edges, wear, edgeCO[secondBiggestClusterIdx]

def biggestContour(mask: np.ndarray):
    """
    Gets biggest contour in a black (0) and white (255) mask 

    Parameters
    ----------
    mask : np.ndarray
        only black or white.

    Returns
    -------
    np.ndarray:
        Coordinates of the biggest contour [x, y].
    int:
        Area of the biggest contour

    """
    # find contour of the mask created by the model
    contourXY, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    maxArea = 0
    maxIndex = 0
    for i in range(len(contourXY)):
        area = cv2.contourArea(contour=contourXY[i])
        if area > maxArea: # get biggest contour
            maxArea = area
            maxIndex = i

    return(contourXY[maxIndex])

def wearWidth(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Function will determine the width of the wear along the main edge

    Parameters
    ----------
    img : np.ndarray
        DESCRIPTION.
    mask : np.ndarray
        Array with the dimension of the image with 0: black (Background) and 1: white (Wear).

    Returns
    -------
    None.

    """
    global p1, p2, p3, p4, boundX, boundY
    global wearArea, uniqueX, distMax

    maxEdgeCO, edgeImg, wearImg, sndEdgeCO = get_edges(img) # biggest edge clusters
    maxEdgeX = maxEdgeCO[:,0]
    maxEdgeY = maxEdgeCO[:,1]
    contourWear, hierarchyWear = cv2.findContours(image=edgeImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image=img, contours=contourWear, contourIdx=0, color=(255,255,255), thickness=3)

    # detect main edge (Linear Regression or Lasso)
    biggestEdge = LinearRegression()    # implement regression algorithm
    # biggestEdge = Lasso()
    biggestEdge.fit(X=maxEdgeX.reshape(-1,1), y=maxEdgeY.reshape(-1,1))
    
    xMinBiggest = min(maxEdgeX) # minimum point of biggest contour
    xMaxBiggest = max(maxEdgeX)
    p1 = [xMinBiggest, biggestEdge.predict([[xMinBiggest]]).ravel()]
    p2 = [xMaxBiggest, biggestEdge.predict([[xMaxBiggest]]).ravel()]


    # detect secondary edge
    sndEdgeX = sndEdgeCO[:,0]
    sndEdgeY = sndEdgeCO[:,1]
    sndEdge = LinearRegression()
    sndEdge.fit(X=sndEdgeX.reshape(-1,1), y=sndEdgeY.reshape(-1,1))
    xMinSecond = min(sndEdgeX) # minimum point of biggest contour
    xMaxSecond = max(sndEdgeX)
    p3 = [xMinSecond, sndEdge.predict([[xMinSecond]]).ravel()]
    p4 = [xMaxSecond, sndEdge.predict([[xMaxSecond]]).ravel()]

    # find wear and wear boundaries
    boundXY = biggestContour(mask=mask)
    boundX = boundXY[:, 0, 1]
    boundY = boundXY[:, 0, 0]

    pointsBelowIdx = boundY < biggestEdge.predict(boundX.reshape(-1,1)).ravel()
    pointsAboveIdx = boundY > biggestEdge.predict(boundX.reshape(-1,1)).ravel()

    # determine area above Regression Line
    boundary_above = np.copy(boundXY)
    if np.any(pointsBelowIdx):    
        boundary_above[pointsBelowIdx, 0,1] = biggestEdge.predict(boundX[pointsBelowIdx].reshape(-1,1)).ravel()
    area_above = cv2.contourArea(boundary_above)
    print("\nArea of Wear (above Regression Line): ", area_above)
    

    # determine area below Regression Line
    boundary_below = np.copy(boundXY)
    if np.any(pointsAboveIdx):
        boundary_below[pointsAboveIdx, 0, 1] = biggestEdge.predict(boundX[pointsAboveIdx].reshape(-1,1)).ravel()
    area_below = cv2.contourArea(boundary_below)
    print("\nArea of wear (below Regression Line): ", area_below)
    
    #Choose boundary with bigger area
    if area_above > area_below:
        boundXY = boundary_above
    else:
        boundXY = boundary_below
    boundX = boundXY[:, 0, 1]
    boundY = boundXY[:, 0, 0]
    
    wearArea = area_above+area_below
    print('\nTotal Wear: ',wearArea)
    # PLOT IMAGE

    #calculate wear width as distance from boundary point to regression line
    #https://de.wikipedia.org/wiki/Abstand#Abstand_zwischen_Punkt_und_Gerade_2
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0]*p1[1] - p1[0]-p2[1]
    distances = np.absolute(a*boundX+b*boundY+c*np.ones(len(boundXY)))*(1/np.sqrt(a**2+b**2))
    # print(boundX)
    uniqueX = np.unique(boundX)
    distMax = np.zeros(len(uniqueX))
    i=0
    for x in uniqueX:
        distMax[i] = max(distances[boundX==(x*np.ones(len(boundX), dtype=np.uint16))])
        i+=1
    # PLOT DISTANCES

    # displayResults(img, edgeImg, wearImg)
        
    return edgeImg, wearImg

def applyMask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    maskHeight = mask.shape[0], maskWidth = mask.shape[1]
    blackMask = np.zeros((maskHeight, maskWidth), dtype=np.unit8)
    maskedImg = cv2.bitwise_or(src1=blackMask, mask=mask)

    return maskedImg
