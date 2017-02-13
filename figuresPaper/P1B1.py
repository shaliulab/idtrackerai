# Import standard libraries
import os
from os.path import isdir, isfile
import sys
import numpy as np
import cPickle as pickle

# Import third party libraries
from pprint import pprint

# Import application/library specifics
sys.path.append('../utils')
sys.path.append('../CNN')

from idTrainer import *
from input_data_cnn import *

def computeDataP1B1(P1B1Dict):
    IMDBname = P1B1Dict['IMDBname']

    databaseInfo, images, labels, imsize, numIndivImdb, numImagesPerIndiv = loadIMDB(IMDBname)
    print images.shape
    # for numIndiv in P1B1Dict['numIndivList']:
    #     for rep in range(P1B1Dict['numRepetitions']):
    #



    return P1B1Dict


if __name__ == '__main__':

    P1B1Dict = {
            'IMDBname': '36dpf_60indiv_29754ImPerInd_curvaturePortrait', # the IMDB is assumed to be in the folder data
            'numIndivList': [2,3],
            'numRepetitions': 2
                }

    P1B1Dict = computeDataP1B1(P1B1Dict)
    print '\n'
    pprint(P1B1Dict)
