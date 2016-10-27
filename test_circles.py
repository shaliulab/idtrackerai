import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer.canvastools import RectangleTool, PaintTool
from skimage.viewer import ImageViewer
from scipy import ndimage
#
cap = cv2.VideoCapture('./data/library/group_2_camera_2/group_2_camera_2_20160508T094727_13.avi')
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image',frame)

plt.ion()
f, ax = plt.subplots()
ax.imshow(frame, interpolation='nearest', cmap='gray')
props = {'facecolor': '#000070',
         'edgecolor': 'white',
         'alpha': 0.3}
rect_tool = RectangleTool(ax, rect_props=props)


plt.show()
numROIs = 4
counter = 0
ROIsCoords = []
centers = []
ROIsShapes = []
mask = np.ones_like(frame,dtype='uint8')*255
while counter < numROIs:
    ROIshape = raw_input('ROI shape (r/c/p)? (press enter after selection)')

    if ROIshape == 'r' or ROIshape == 'c':
        ROIsShapes.append(ROIshape)

        rect_tool.callback_on_enter(rect_tool.extents)
        coord = np.asarray(rect_tool.extents).astype('int')

        print 'ROI coords, ', coord
        goodROI=raw_input('Is the selection correct? [y]/n: ')

        if goodROI == 'y':
            ROIsCoords.append(coord)
            if ROIshape == 'r':
                cv2.rectangle(mask,(coord[0],coord[2]),(coord[1],coord[3]),0,-1)
                centers.append(None)
            if ROIshape == 'c':
                center = ((coord[1]+coord[0])/2,(coord[3]+coord[2])/2)
                angle = 0
                axes = tuple(sorted(((coord[1]-coord[0])/2,(coord[3]-coord[2])/2)))
                print center, angle, axes
                cv2.ellipse(mask,center,axes,angle,0,360,0,-1)
                centers.append(center)

    counter = len(ROIsCoords)

print 'centers, ', centers





    # elif ROIshape == 'p':
    #     p = PaintTool(ax,np.shape(frame[:-1]),10,0.2)
    #     # plt.show()
    #     mask_p = p.overlay
    #     mask_p = ndimage.morphology.binary_fill_holes(mask)
    #
    #     print 'ROI coords, ', coord
    #     goodROI=raw_input('Is the selection correct? [y]/n: ')
    #     if goodROI == 'y':
    #         mask = mask + mask_p
    #         print mask



maskedFrame = cv2.addWeighted(frame,1,mask,1,0)
cv2.imshow('frameMasked',maskedFrame)
cv2.waitKey()

# def get_rect_coord(extents):
#     global viewer,coord_list
#     coord_list.append(extents)
#
# def get_ROI(im):
#     global viewer,coord_list
#
#     selecting=True
#     while selecting:
#         f, ax = plt.subplots()
#         ax.imshow(im, interpolation='nearest', cmap='gray')
#         coord_list = []
#         rect_tool = RectangleTool(ax, on_enter=get_rect_coord)
#         print "Draw your selections, press ENTER to validate one and close the window when you are finished"
#         plt.show()
#         finished=raw_input('Is the selection correct? [y]/n: ')
#         if finished!='n':
#             selecting=False
#     return coord_list
#
# a = get_ROI(frame)
# print a
