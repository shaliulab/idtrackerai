# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)


from __future__ import division
import sys
import numpy as np
import scipy.ndimage
from scipy.signal import argrelmax
import cv2
sys.path.append('./')
from idtrackerai.constants import SMOOTH_SIGMA, HEAD_DIAMETER
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.fishcontour")

def find_max(curv,n=2):
    """Returns the nth local maximum of the array f (Default is 2nd)
    :param f: one dimensional array
    :param n: positive integer
    """
    (max_i,) = argrelmax(curv, mode='wrap')
    b_s = sorted(max_i.tolist(), key=lambda x:curv[x])
    try:
        nose = b_s[-n]
        return nose
    except IndexError:
        logger.warn("Warning, no nose detected in a frame. The maximum is returned instead")
        return b_s[-1]

class FishContour():
    def __init__(self,curve):
        """It expects a (N,2) numpy array, with (:,0) the x coordinates and (:,1) the y coordinates
        """
        self.c = curve

    @classmethod
    def fromcv2contour(cls,cv2contour):
        """Creates FishContour instance from a cv2 style single contour
        """
        return cls(cv2contour[:,0,:].astype(np.float32))

    def __str__(self):
        return self.c.__str__()

    def derivative(self):
        return scipy.ndimage.convolve1d(self.c, [-0.5,0.0,0.5],mode = 'wrap',axis=0)
    def second_derivative(self):
        return scipy.ndimage.convolve1d(self.c , [1.0,-2.0,1.0], mode='wrap',axis=0)
    def curvature(self):
        """ Calculates signed curvature of the planar curve.
        """
        [x_1,y_1] = self.derivative().transpose()
        [x_2,y_2] = self.second_derivative().transpose()
        return (x_1*y_2 - y_1*x_2)/np.power(x_1*x_1 + y_1*y_1,3/2)

    def find_nose(self):
        """It returns coordinates of nose in the contour
        It finds it by calculating a smoother version of the contour
        but with the same indexing and then finding the second maximum
        (in abs value) of the curvatue (first max is usually the tail)
        """
        smoother = smooth(self)
        nose_i = find_max(abs(smoother.curvature()),n=2)
        return self.c[nose_i,:]

    def find_nose_and_orientation(self,head_size = HEAD_DIAMETER):
        """It returns nose coordinates, angle needed to rotate so nose points to negative y
        and the centroid of the head

        head_size gives the distance between the nose and the base of the head
        """
        nose = self.find_nose()#.astype(np.int32)
        distance = np.power(self.c[:,0]-nose[0],2) + np.power(self.c[:,1]-nose[1],2)
        head_centroid = FishContour(self.c[np.where(distance < head_size*head_size)]).centroid()
        orvec = nose - head_centroid
        angle = np.degrees(np.arctan2(orvec[1],orvec[0]))
        return nose,angle+90,np.array(head_centroid)

    def ascvcontour(self):
        """Returns a contour in opencv style, i.e. (x,0,y).
        It is needed in cv2.moments() in OpenCV 2, in version 3 not really
        """
        return np.expand_dims(self.c,axis=1)

    def centroid(self):
        """Returns the centroid of the contour
        """
        M = cv2.moments(self.ascvcontour())
        try:
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
        except:
            cX = 0
            cY = 0
            print('Warning: M["m00"] might be zero')
        return (cX,cY)

def smooth(contour):
    """Returns a smoother FishContour() with the same length.
    Care is taken to make indeces invariant (i.e. no offset caused by filtering)

    :param contour: instance of FishContour()
    """
    return FishContour(scipy.ndimage.filters.gaussian_filter1d(contour.c, SMOOTH_SIGMA, mode='wrap',axis=0))
