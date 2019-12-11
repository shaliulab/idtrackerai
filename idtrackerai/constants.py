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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import sys
import os

MIN_FLOAT = sys.float_info[3]
MAX_FLOAT = sys.float_info[0]

IDTRACKERAI_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEST_VIDEO_URL='https://drive.google.com/uc?id=1uBOEMGxrOed8du7J9Rt-dlXdqOyhCpMC'

PROCESSES = ['preprocessing','protocols1_and_2', 'protocol3_pretraining',
            'protocol3_accumulation', 'residual_identification',
            'post_processing']

#######################################
##########       video      ###########
#######################################
'''***AVAILABLE_VIDEO_EXTENSION***
Tested and available video extensions
'''
AVAILABLE_VIDEO_EXTENSION = ['.avi', '.AVI', '.mp4', '.MP4', '.mpg', '.MPG', '.mov', '.MOV']
'''***FRAMES_PER_EPISODE***
Number of frames per video chunk. Used to parallelise processes
'''
FRAMES_PER_EPISODE = int(os.environ.get('FRAMES_PER_EPISODE', 500))
#######################################
##########   preprocessing  ###########
#######################################
"""***NUMBER_OF_ANIMALS_DEFAULT
Number of animals to be tracked
"""
NUMBER_OF_ANIMALS_DEFAULT = int(os.environ.get('FRAMES_PER_EPISODE', 8))
"""***MIN_AREA_LOWER, MIN_AREA_UPPER***
Lower and upper bounds for the minimum area slider
"""
MIN_AREA_LOWER, MIN_AREA_UPPER = 0, 10000
"""***DEFAULT_RESOLUTION_REDUCTION***
Ratio to which the width and height are rescaled.
"""
RES_REDUCTION_DEFAULT = float(os.environ.get('RES_REDUCTION_DEFAULT', 1.0))
"""***MIN_AREA_DEFAULT, MAX_AREA_DEFAULT***
Default value for min area in preprocessing
"""
MIN_AREA_DEFAULT, MAX_AREA_DEFAULT = 150, 60000
"""***AREA_LOWER, AREA_UPPER***
Lower and upper bounds for the maximum area slider
"""
AREA_LOWER, AREA_UPPER = 0, 60000
"""***MIN_THRESHOLD, AREA_UPPER***
Lower and upper bounds for the maximum area slider
"""
MIN_THRESHOLD, MAX_THRESHOLD = 0, 255
"""***MIN_THRESHOLD_DEFAULT, MAX_THRESHOLD_DEFAULT***
Default value for min area in preprocessing
"""
MIN_THRESHOLD_DEFAULT, MAX_THRESHOLD_DEFAULT = 0, 155
"""***VEL_PERCENTILE***
percentile on the average speed of the individuals used to compute the maximal
velocity threshold
"""
VEL_PERCENTILE = int(os.environ.get('VEL_PERCENTILE', 99))
'''***STD_TOLERANCE***
Tolerance coefficient in the computation of the individual model area.
'''
STD_TOLERANCE = int(os.environ.get('STD_TOLERANCE', 4))
"""***BACKGROUND_SUBTRACTION_PERIOD***
Period used to sample the video to compute the background model
"""
BACKGROUND_SUBTRACTION_PERIOD = int(os.environ.get('BACKGROUND_SUBTRACTION_PERIOD', 500))
"""***NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION***
Number of jobs used to compute the background model
"""
NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION = -2 # Set None to use the default mode of the system. (see segmentation_utils.py module for details)
"""***NUMBER_OF_JOBS_FOR_SEGMENTATION***
Number of jobs used to perform the segmentation
"""
NUMBER_OF_JOBS_FOR_SEGMENTATION = -2 # Set None to use the default mode of the system. (see segmentation.py module for details)
"""***SIGMA_GAUSSIAN_BLURRING***
sigma for gaussian blurring tests
"""
SIGMA_GAUSSIAN_BLURRING = None
"""***SAVE_PIXELS***
Where to store the pixels list of every blob
DISK: pixels are stored in hdf5 files in the disk
RAM: pixels are stored in the blob object in the ram
NOT: pixels are not stored and are computed from the contour everytime
"""
SAVE_PIXELS = 'DISK' # 'RAM' or 'NONE'
"""***SAVE_SEGMENTATION_IMAGE***
Where to store the segmentation image of every blob
DISK: image is stored in hdf5 files in the disk
RAM: image is stored in the blob object in the ram
NOT: image is not stored and are computed from the contour everytime
"""
SAVE_SEGMENTATION_IMAGE = 'DISK' # 'RAM' or 'NONE'
"""***PLOT_CROSSING_DETECTOR***
"""
PLOT_CROSSING_DETECTOR=False
#######################################
#########  global fragments  ##########
#######################################
'''***MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION***
Minimum number of frame to allow an individual fragment to be part of a
global one
'''
MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = int(os.environ.get('MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION', 3))
#######################################
##########        CNN       ###########
#######################################
'''***PLOT_ACCUMULATION_STEPS***
'''
PLOT_ACCUMULATION_STEPS = False
'''***CNN_MODEL***
Architecture for the IDCNN network. By default the model 0 is used, which correspond
to the model of the paper. Check the different models in the module cnn_architectures.py
in the folder network and how they are map to a single value code in the module
id_CNN.py in the network/identification_model folder.
'''
CNN_MODEL = 0
'''***KNOWLEDGE_TRANSFER_FOLDER_IDCNN***
Folder for a accumulation folder with a model from another video. Note that if
the IDENTITY_TRANSFER flag is True, then the IDENTIFICATION_IMAGE_SIZE will be
taken from the knowledge_transfer_info_dict.
'''
KNOWLEDGE_TRANSFER_FOLDER_IDCNN = None
'''***IDENTITY_TRANSFER***
Bloonean
'''
IDENTITY_TRANSFER = False
"""***IDENTIFICATION_IMAGE_SIZE***
size of the identification images. Used for idmatcher.ai
"""
IDENTIFICATION_IMAGE_SIZE = None #It should be a tuple of len 3 (width, height, chanels), e.g. (46, 46, 1)
'''***USE_ADAM_OPTIMISER***
By default the identification neural network is trained with SGD. If this parameter
is set to TRUE, the network will be trained with the ADAM optimizer.
'''
USE_ADAM_OPTIMISER = False
'''***LAYERS_TO_OPTIMISE_ACCUMULATION
List of layers to be optimized. None indicates that all layers are optimizsed.
To check the name of the different layers see the module cnn_architectures.py
in the folder network.
'''
LAYERS_TO_OPTIMISE_ACCUMULATION = None # ['fully-connected1','fully_connected_pre_softmax']
LAYERS_TO_OPTIMISE_PRETRAINING = None
'''***LEARNING_RATE_IDCNN_ACCUMULATION***
'''
LEARNING_RATE_IDCNN_ACCUMULATION = float(os.environ.get('LEARNING_RATE_IDCNN_PRETRAINING', 0.005))
'''***LEARNING_RATE_IDCNN_PRETRAINING***
'''
LEARNING_RATE_IDCNN_PRETRAINING = float(os.environ.get('LEARNING_RATE_IDCNN_PRETRAINING', 0.01))
'''***LEARNING_RATE_DCD***
'''
LEARNING_RATE_DCD = float(os.environ.get('LEARNING_RATE_DCD', 0.001))
'''***KEEP_PROB_IDCNN_ACCUMULATION***
'''
KEEP_PROB_IDCNN_ACCUMULATION = float(os.environ.get('KEEP_PROB_IDCNN_ACCUMULATION', 1.))
'''***KEEP_PROB_IDCNN_PRETRAINING***
'''
KEEP_PROB_IDCNN_PRETRAINING = float(os.environ.get('KEEP_PROB_IDCNN_PRETRAINING', 1.))
'''***KEEP_PROB_DCD***
'''
KEEP_PROB_DCD = float(os.environ.get('KEEP_PROB_DCD', 1.))
'''***VALIDATION_PROPORTION***
Protortion of images used for validation in the IDCNN model
'''
VALIDATION_PROPORTION = float(os.environ.get('VALIDATION_PROPORTION', .1))
'''***BATCH_SIZE_DCD, BATCH_SIZE_IDCNN***
size of the batches used to train the DCD and idCNN, respectively.
'''
BATCH_SIZE_IDCNN = int(os.environ.get('BATCH_SIZE_IDCNN', 50))
BATCH_SIZE_DCD = int(os.environ.get('BATCH_SIZE_DCD', 50))
'''***BATCH_SIZE_PREDICTIONS***
size of the batches used to get the output from the DCD and idCNN, respectively.
Remark: This is done to prevent out-of-memory error in the GPU
'''
BATCH_SIZE_PREDICTIONS_DCD = int(os.environ.get('BATCH_SIZE_PREDICTIONS_DCD', 100))
BATCH_SIZE_PREDICTIONS_IDCNN = int(os.environ.get('BATCH_SIZE_PREDICTIONS_IDCNN', 500))
'''***LEARNING_PERCENTAGE_DIFFERENCE_1 ***
Overfitting threshold during training
'''
LEARNING_PERCENTAGE_DIFFERENCE_1_DCD = float(os.environ.get('LEARNING_PERCENTAGE_DIFFERENCE_1_DCD', .005))
LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN = float(os.environ.get('LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN', .005))
'''***LEARNING_PERCENTAGE_DIFFERENCE_2
Loss plateau threshold during training
***'''
LEARNING_PERCENTAGE_DIFFERENCE_2_DCD = float(os.environ.get('LEARNING_PERCENTAGE_DIFFERENCE_2_DCD', .005))
LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN = float(os.environ.get('LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN', .005))
'''***OVERFITTING_COUNTER_THRESHOLD ***
Number of consecutive overfitting epochs in order to stop the training
'''
OVERFITTING_COUNTER_THRESHOLD_DCD = int(os.environ.get('OVERFITTING_COUNTER_THRESHOLD_DCD', 5))
OVERFITTING_COUNTER_THRESHOLD_IDCNN = int(os.environ.get('OVERFITTING_COUNTER_THRESHOLD_IDCNN', 5))
OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM = int(os.environ.get('OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM', 10))
'''***MAXIMUM_NUMBER_OF_EPOCHS ***
Maximum number of epochs before forcing the training to stop
'''
MAXIMUM_NUMBER_OF_EPOCHS_DCD = int(os.environ.get('MAXIMUM_NUMBER_OF_EPOCHS_DCD', 100))
MAXIMUM_NUMBER_OF_EPOCHS_IDCNN = int(os.environ.get('MAXIMUM_NUMBER_OF_EPOCHS_IDCNN', 10000))
'''***KEEP_PROB***
Default dropout in fully-connected layers if the CNN models (can be changed
when instantiating the parameter class to be init the CNN)
'''
KEEP_PROB = float(os.environ.get('KEEP_PROB', 1.0))
'''***MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR***
Minimum number of crossings required to train the crossing detector, otherwise
only the model area is used to distinguish crossings from individuals
'''
MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR = int(os.environ.get('MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR', 10))
#######################################
# Deep fingerprint protocols cascade  #
#######################################
'''***THRESHOLD_EARLY_STOP_ACCUMULATION***
If the total of accumulated images + images to be accumulated is above this
threshold, we stop the accumulation (holds for each deep fingerprint protocol)
'''
THRESHOLD_EARLY_STOP_ACCUMULATION = float(os.environ.get('THRESHOLD_EARLY_STOP_ACCUMULATION', .9995))
""" ***THRESHOLD_ACCEPTABLE_ACCUMULATION***
During the deep fingerprint protocol cascade we evaluate the information
contained in global fragments and, if necessary (1), we accumulate references.
An accumulation if considered acceptable if at the end of the process the
ratio (number of accumulated image) / (number of image potentially accumulable)
if equal or bigger than THRESHOLD_ACCEPTABLE_ACCUMULATION.
----
(1) THRESHOLD_EARLY_STOP_ACCUMULATION is not reached
"""
THRESHOLD_ACCEPTABLE_ACCUMULATION = float(os.environ.get('THRESHOLD_ACCEPTABLE_ACCUMULATION', .9))
"""***MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
***"""
MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS = int(os.environ.get('MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS', 3))
'''***MAXIMAL_IMAGES_PER_ANIMAL***
Maximal number of images per indiviudal to be included in the training dataset
of the idCNN
'''
MAXIMAL_IMAGES_PER_ANIMAL = int(os.environ.get('MAXIMAL_IMAGES_PER_ANIMAL', 3000))
'''***RATIO_OLD***
Percentage of the MAXIMAL_IMAGES_PER_ANIMAL to be subsampled from the images
accumulated in the previous steps
'''
RATIO_OLD = float(os.environ.get('RATIO_OLD', 0.6))
'''***RATIO_NEW***
Percentage of the MAXIMAL_IMAGES_PER_ANIMAL to be subsampled from the images
accumulated in the current step
'''
RATIO_NEW = float(os.environ.get('RATIO_NEW', 0.4))
'''***CERTAINTY_THRESHOLD***
Minimal certainty of the assignment of an individual fragment to be considered
acceptable
'''
CERTAINTY_THRESHOLD = float(os.environ.get('CERTAINTY_THRESHOLD', .1))
'''***MAX_RATIO_OF_PRETRAINED_IMAGES***
Maximum ratio of accumulable images to stop protocol 3 (pretraining)
'''
MAX_RATIO_OF_PRETRAINED_IMAGES = float(os.environ.get('MAX_RATIO_OF_PRETRAINED_IMAGES', .95))
'''***MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION***
Minimal ratio of accumulated images over accumulable images to allow the
partial accumulation strategy to start
'''
MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION = float(os.environ.get('MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION', .5))
#######################################
########   post-processing   ##########
#######################################
'''***FIXED_IDENTITY_THRESHOLD***
If the certainty of the assignment of an individual fragment is above this
threshold we consider the identification certain. Thus, it won't be modified
either during the final identification or post-processing
'''
FIXED_IDENTITY_THRESHOLD = float(os.environ.get('FIXED_IDENTITY_THRESHOLD', .9))
#######################################
##########   fish-specific  ###########
#######################################
'''***SMOOTH_SIGMA***
Parameter giving the standard deviation of the gaussian filtering in the
contour to calculate curvature
'''
SMOOTH_SIGMA = int(os.environ.get('SMOOTH_SIGMA', 10))
'''***HEAD_DIAMETER***
Distance between nose and base of the head
'''
HEAD_DIAMETER = int(os.environ.get('HEAD_DIAMETER', 20))
##################################################
##########   data management policies  ###########
##################################################
""" Data policy to be applied at the end of the tracking
'all': saves all the data as it is generated from tracking
'trajectories': saves only the trajectories
'validation': saves the information needed to validate the video
'knowledge_transfer': saves the information needed to perfom identity_transfer to another video
'idmatcher.ai': saves the information needed to perform identity_matching between videos
"""
DATA_POLICY = os.environ.get('DATA_POLICY', 'all')


################################################
##########       postprocessing      ###########
################################################

INDIVIDUAL_VIDEO_WIDTH_HEIGHT = None

CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON = False
