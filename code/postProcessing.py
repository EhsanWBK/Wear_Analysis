''' ## Post-Processing of Segmentation

NOT WORKING AT THE MOMENT!

Evaluate segmentation and create wear boundaries.

Accessible Functions:
- 
'''

import matplotlib.pyplot as plt
from numpy import column_stack, ndarray
from scipy.spatial import ConvexHull

def wearBoundaries(img: ndarray, mask: ndarray):
    # determine biggest edge; split contours into two; determine biggest (other one is residue);
    # let bigger contour part be real boundary
    # for milling: determine biggest edge; create box around

    # boundCOS = biggestContour(mask=mask)
    # boundX = boundCOS[:,0,1]
    # boundY = boundCOS[:,0,0]
    return

def displayEval(image: ndarray, mask: ndarray):
    # display original image, predicted mask, wear boundaries, 
    # wear parameters (in original image), list of wear parameters
    # print out wear parameters

    # original image, predicted mask, and wear boundaries
    plt.subplot(321), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(322), plt.imshow(mask, cmap='gray'), plt.title('Predicted Mask')
    plt.subplot(323), plt.imshow(), plt.title('Wear Boundaries')

    # plot wear parameters
    plt.subplot(324)
    plt.subplot(325)
    plt.subplot(326)

    return

def segmentationEval():
    # distiction between milling and turning
    # wear area, wear width
    # display results

    return

# ----------------------------------------------------
# --------------------- DUMPSTER ---------------------
# ----------------------------------------------------
# delete everything below once done

def displayResults(originalImg, edgesImg, wearImg):
    global p1, p2, p3, p4, boundX, boundY
    global wearArea, uniqueX, distMax, curMask
    # plot original image
    plt.subplot(221), plt.imshow(originalImg,cmap='gray'), plt.title('Original Image')
    
    # plot boundaries and biggest edge
    plt.subplot(222), plt.imshow(originalImg, cmap='gray'), plt.title('Biggest Edges')
    plt.plot(boundY, boundX, 'r+',label='Boundary')
    # plt.plot([p1[0],p2[0]], [p1[1], p2[1]], label='Biggest Tool edge')
    # plt.plot([p3[0],p4[0]], [p3[1], p4[1]], label='Second Biggest Tool edge')
    
    # plot wear width and distance
    plt.subplot(223), plt.plot(uniqueX, distMax), plt.title('Wear Width')
    plt.xlabel('X-Coordinate of Boundary Point'), plt.ylabel('Width of Wear at given point')
    
    # plot image with mask
    plt.subplot(224), plt.imshow(originalImg, cmap='gray'), plt.title('Wear Image')
    boundKos = column_stack((boundY, boundX))
    hullBound = ConvexHull(boundKos)
    plt.fill(boundKos[hullBound.vertices,0],boundKos[hullBound.vertices,1],'white')
    plt.xticks([]), plt.yticks([]), plt.show()

    # save last image to temporary file
    fig1, ax1 = plt.subplots()
    ax1.imshow(originalImg, cmap='gray')
    ax1.set_title('WearImage')
    ax1.fill(boundKos[hullBound.vertices,0],boundKos[hullBound.vertices,1],'white')
    fig1.savefig('temp.png')
