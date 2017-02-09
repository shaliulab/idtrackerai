from pylab import *
from skimage import data
from skimage.viewer.canvastools import RectangleTool
from skimage.viewer import ImageViewer
from skimage.draw import polygon

import sys
sys.path.append('../utils')
sys.path.append('../preprocessing')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *



def get_rect_coord(extents):
    global viewer,coord_list, coord_shape
    coord_list.append(extents)
    coord_shape.append(getInput('Select ROI shape', 'type r for rectangular ROI and c for circular'))

def get_ROI(im):
    global viewer,coord_list, coord_shape

    selecting=True
    while selecting:
        viewer = ImageViewer(im)
        coord_list = []
        coord_shape = []
        rect_tool = RectangleTool(viewer, on_enter=get_rect_coord)

        print "Draw a ROI, press ENTER to validate and repeat the procedure to draw more than one ROI. Close the window when you are finished"

        viewer.show()

        finished=getInput('Confirm selection','Is the selection correct? [y]/n: ')
        if finished!='n':
            selecting = False
        else:
            get_ROI(im)
    return coord_list, coord_shape
