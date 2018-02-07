from __future__ import absolute_import, print_function, division
import sys

MIN_FLOAT = sys.float_info[3]
MAX_FLOAT = sys.float_info[0]

#######################################
##########       video      ###########
#######################################
AVAILABLE_VIDEO_EXTENSION = ['.avi', '.mp4', '.mpg', '.MOV']
'''***FRAMES_PER_EPISODE***
Number of frames per video chunk. Used to parallelise processes
'''
FRAMES_PER_EPISODE = 100

#######################################
##########   preprocessing  ###########
#######################################
"""***VEL_PERCENTILE***
percentile on the average speed of the individuals used to compute the maximal
velocity threshold
"""
VEL_PERCENTILE = 99
'''***STD_TOLERANCE***
Tolerance coefficient in the computation of the individual model area.
'''
STD_TOLERANCE = 4
"""***BACKGROUND_SUBTRACTION_PERIOD***
Period used to sample the video to compute the background model
"""
BACKGROUND_SUBTRACTION_PERIOD = 100

#######################################
#########  global fragments  ##########
#######################################
'''***MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION***
Minimum number of frame to allow an individual fragment to be part of a
global one
'''
MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = 3

#######################################
##########        CNN       ###########
#######################################
'''***BATCH_SIZE_DCD, BATCH_SIZE_IDCNN***
size of the batches used to train the DCD and idCNN, respectively.
'''
BATCH_SIZE_IDCNN = 50
BATCH_SIZE_DCD = 50
'''***BATCH_SIZE_PREDICTIONS***
size of the batches used to get the output from the DCD and idCNN, respectively.
Remark: This is done to prevent out-of-memory error in the GPU
'''
BATCH_SIZE_PREDICTIONS_DCD = 100
BATCH_SIZE_PREDICTIONS_IDCNN = 500

'''***LEARNING_PERCENTAGE_DIFFERENCE_1 ***
Overfitting threshold during training
'''
LEARNING_PERCENTAGE_DIFFERENCE_1_DCD = .005
LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN = .005
'''***LEARNING_PERCENTAGE_DIFFERENCE_2
Loss plateau threshold during training
***'''
LEARNING_PERCENTAGE_DIFFERENCE_2_DCD = .005
LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN = .005
'''***OVERFITTING_COUNTER_THRESHOLD ***
Number of consecutive overfitting epochs in order to stop the training
'''
OVERFITTING_COUNTER_THRESHOLD_DCD = 5
OVERFITTING_COUNTER_THRESHOLD_IDCNN = 5
'''***MAXIMUM_NUMBER_OF_EPOCHS ***
Maximum number of epochs before forcing the training to stop
'''
MAXIMUM_NUMBER_OF_EPOCHS_DCD = 100
MAXIMUM_NUMBER_OF_EPOCHS_IDCNN = 10000
'''***KMEANS_NUMBER_OF_STEPS_EMBEDDING_EXPLORATION ***
Number of KM iterations when clustering the embedding generated by the
first fully-connected layer of the IDCNN
'''
KMEANS_NUMBER_OF_STEPS_EMBEDDING_EXPLORATION_IDCNN = 100
'''***KEEP_PROB***
Default dropout in fully-connected layers if the CNN models (can be changed
when instantiating the parameter class to be init the CNN)
'''
KEEP_PROB = 1.0

#######################################
# Deep fingerprint protocols cascade  #
#######################################
'''***THRESHOLD_EARLY_STOP_ACCUMULATION***
If the total of accumulated images + images to be accumulated is above this
threshold, we stop the accumulation (holds for each deep fingerprint protocol)
'''
THRESHOLD_EARLY_STOP_ACCUMULATION = .9995
""" ***THRESHOLD_ACCEPTABLE_ACCUMULATION***
During the deep fingerprint protocol cascade we evaluate the information
contained in global fragments and, if necessary (1), we accumulate references.
An accumulation if considered acceptable if at the end of the process the
ratio (number of accumulated image) / (number of image potentially accumulable)
if equal or bigger than THRESHOLD_ACCEPTABLE_ACCUMULATION.
----
(1) THRESHOLD_EARLY_STOP_ACCUMULATION is not reached
"""
THRESHOLD_ACCEPTABLE_ACCUMULATION = .9

"""***MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
***"""
MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS = 3

'''***MAXIMAL_IMAGES_PER_ANIMAL***
Maximal number of images per indiviudal to be included in the training dataset
of the idCNN
'''
MAXIMAL_IMAGES_PER_ANIMAL = 3000
'''***RATIO_OLD***
Percentage of the MAXIMAL_IMAGES_PER_ANIMAL to be subsampled from the images
accumulated in the previous steps
'''
RATIO_OLD = 0.6
'''***RATIO_NEW***
Percentage of the MAXIMAL_IMAGES_PER_ANIMAL to be subsampled from the images
accumulated in the current step
'''
RATIO_NEW = 0.4
'''***CERTAINTY_THRESHOLD***
Minimal certainty of the assignment of an individual fragment to be considered
acceptable
'''
CERTAINTY_THRESHOLD = .1
'''***MAX_RATIO_OF_PRETRAINED_IMAGES***
Maximum ratio of accumulable images to stop protocol 3 (pretraining)
'''
MAX_RATIO_OF_PRETRAINED_IMAGES = .95
'''***MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION***
Minimal ratio of accumulated images over accumulable images to allow the
partial accumulation strategy to start
'''
MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION = .5
'''***RESTORE_CRITERION***
Values: {"last", "best"}
When stopping the accumulation it is possible either to store the weights of
the last model or the ones belonging to the model realising the minimum loss in
validation.
'''
RESTORE_CRITERION = 'last'

#######################################
########   post-processing   ##########
#######################################
'''***FIXED_IDENTITY_THRESHOLD***
If the certainty of the assignment of an individual fragment is above this
threshold we consider the identification certain. Thus, it won't be modified
either during the final identification or post-processing
'''
FIXED_IDENTITY_THRESHOLD = .9

#######################################
##########   fish-specific  ###########
#######################################
'''***SMOOTH_SIGMA***
Parameter giving the standard deviation of the gaussian filtering in the
contour to calculate curvature
'''
SMOOTH_SIGMA = 10
'''***HEAD_DIAMETER***
Distance between nose and base of the head
'''
HEAD_DIAMETER = 20
