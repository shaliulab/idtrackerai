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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys

MAX_FLOAT = sys.float_info[0]

IDTRACKERAI_FOLDER = os.path.dirname(os.path.abspath(__file__))
TEST_VIDEO_URL = (
    "https://drive.google.com/uc?id=1uBOEMGxrOed8du7J9Rt-dlXdqOyhCpMC"
)
DATA_PATH = os.path.join(IDTRACKERAI_FOLDER, "data")
COMPRESSED_VIDEO_PATH = os.path.join(
    DATA_PATH,
    "example_video_compressed",
    "conflict3and4_20120316T155032_14_compressed.avi",
)
COMPRESSED_VIDEO_PATH_2 = os.path.join(
    DATA_PATH,
    "example_video_compressed",
    "conflict3and4_20120316T155032_13_compressed.avi",
)

#######################################
#### Constants for tests ##############
#######################################
COMPRESSED_VIDEO_NUM_FRAMES = 508
COMPRESSED_VIDEO_NUM_FRAMES_2 = 501
COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES = 1009
COMPRESSED_VIDEO_WIDTH = 1160
COMPRESSED_VIDEO_HEIGHT = 938

#######################################
##########       video      ###########
#######################################
AVAILABLE_VIDEO_EXTENSION = [
    ".avi",
    ".AVI",
    ".mp4",
    ".MP4",
    ".mpg",
    ".MPG",
    ".mov",
    ".MOV",
]
###############################################################################
# Animal detection advanced parameters
###############################################################################
# TODO: Fix compatibility frames per episode and background subtraction period
FRAMES_PER_EPISODE = int(os.environ.get("FRAMES_PER_EPISODE", 500))
BACKGROUND_SUBTRACTION_PERIOD = int(
    os.environ.get("BACKGROUND_SUBTRACTION_PERIOD", 250)
)
BACKGROUND_SUBTRACTION_STAT = os.environ.get(
    "BACKGROUND_SUBTRACTION_STAT", "max"
)
# Set None to use the default mode of the system.
# (see segmentation_utils.py module for details)
NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION = -2
# Set None to use the default mode of the system.
# (see segmentation.py module for details)
NUMBER_OF_JOBS_FOR_SEGMENTATION = -2
SIGMA_GAUSSIAN_BLURRING = None
##################################################
# Data storage advanced parameters
##################################################
# 'all': saves all the data as it is generated from tracking
# 'trajectories': saves only the trajectories
# 'validation': saves the information needed to validate the video
# 'knowledge_transfer': saves the information needed to perfom identity_transfer
# to another video
# 'idmatcher.ai': saves the information needed to perform identity_matching
# between videos
DATA_POLICY = os.environ.get("DATA_POLICY", "all")
SAVE_AREAS = True
# Where to store the pixels list of every blob
# DISK: pixels are stored in hdf5 files in the disk
# RAM: pixels are stored in the blob object in the ram
# NONE: pixels are not stored and are computed from the contour everytime
SAVE_PIXELS = "DISK"  # 'RAM' or 'NONE'
# Where to store the segmentation image of every blob
# DISK: image is stored in hdf5 files in the disk
# RAM: image is stored in the blob object in the ram
# NONE: image is not stored and are computed from the contour everytime
SAVE_SEGMENTATION_IMAGE = "DISK"  # "DISK"  # 'RAM' or 'NONE'

###############################################################################
# GUI advanced parameters
###############################################################################
GUI_MINIMUM_HEIGHT = 500
GUI_MINIMUM_WIDTH = 1000
RES_REDUCTION_DEFAULT = float(os.environ.get("RES_REDUCTION_DEFAULT", 1.0))
NUMBER_OF_ANIMALS_DEFAULT = int(os.environ.get("NUMBER_OF_ANIMALS_DEFAULT", 8))
AREA_LOWER, AREA_UPPER = 0, 60000
MIN_AREA_DEFAULT, MAX_AREA_DEFAULT = 150, 6000
MIN_THRESHOLD, MAX_THRESHOLD = 0, 255
MIN_THRESHOLD_DEFAULT, MAX_THRESHOLD_DEFAULT = 0, 155
EXTRA_PIXELS_BBOX = 45
###############################################################################
# Crossing detector advanced parameters
###############################################################################
# Set None to use the default mode of the system.
# (see segmentation.py module for details)
MODEL_AREA_SD_TOLERANCE = int(os.environ.get("MODEL_AREA_SD_TOLERANCE", 4))
NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES = -2
MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR = int(
    os.environ.get(
        "MINIMUM_NUMBER_OF_CROSSINGS_TO_TRAIN_CROSSING_DETECTOR", 10
    )
)
MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR = int(
    os.environ.get("MAX_IMAGES_PER_CLASS_CROSSING_DETECTOR", 3000)
)
LEARNING_RATE_DCD = float(os.environ.get("LEARNING_RATE_DCD", 0.001))
KEEP_PROB_DCD = float(os.environ.get("KEEP_PROB_DCD", 1.0))
BATCH_SIZE_DCD = int(os.environ.get("BATCH_SIZE_DCD", 50))
BATCH_SIZE_PREDICTIONS_DCD = int(
    os.environ.get("BATCH_SIZE_PREDICTIONS_DCD", 100)
)
LEARNING_PERCENTAGE_DIFFERENCE_1_DCD = float(
    os.environ.get("LEARNING_PERCENTAGE_DIFFERENCE_1_DCD", 0.005)
)
LEARNING_PERCENTAGE_DIFFERENCE_2_DCD = float(
    os.environ.get("LEARNING_PERCENTAGE_DIFFERENCE_2_DCD", 0.005)
)
OVERFITTING_COUNTER_THRESHOLD_DCD = int(
    os.environ.get("OVERFITTING_COUNTER_THRESHOLD_DCD", 5)
)
MAXIMUM_NUMBER_OF_EPOCHS_DCD = int(
    os.environ.get("MAXIMUM_NUMBER_OF_EPOCHS_DCD", 30)
)
###############################################################################
# Tracker with identities
###############################################################################
MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION = int(
    os.environ.get(
        "MINIMUM_NUMBER_OF_FRAMES_TO_BE_A_CANDIDATE_FOR_ACCUMULATION", 3
    )
)
KNOWLEDGE_TRANSFER_FOLDER_IDCNN = None
IDENTITY_TRANSFER = False
LAYERS_TO_OPTIMISE_ACCUMULATION = (
    None  # ['fully-connected1','fully_connected_pre_softmax']
)
LAYERS_TO_OPTIMISE_PRETRAINING = None
LEARNING_RATE_IDCNN_ACCUMULATION = float(
    os.environ.get("LEARNING_RATE_IDCNN_PRETRAINING", 0.005)
)
VALIDATION_PROPORTION = float(os.environ.get("VALIDATION_PROPORTION", 0.1))
BATCH_SIZE_IDCNN = int(os.environ.get("BATCH_SIZE_IDCNN", 50))
BATCH_SIZE_PREDICTIONS_IDCNN = int(
    os.environ.get("BATCH_SIZE_PREDICTIONS_IDCNN", 500)
)
LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN = float(
    os.environ.get("LEARNING_PERCENTAGE_DIFFERENCE_1_IDCNN", 0.005)
)
LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN = float(
    os.environ.get("LEARNING_PERCENTAGE_DIFFERENCE_2_IDCNN", 0.005)
)

OVERFITTING_COUNTER_THRESHOLD_IDCNN = int(
    os.environ.get("OVERFITTING_COUNTER_THRESHOLD_IDCNN", 5)
)
OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM = int(
    os.environ.get("OVERFITTING_COUNTER_THRESHOLD_IDCNN_FIRST_ACCUM", 10)
)

MAXIMUM_NUMBER_OF_EPOCHS_IDCNN = int(
    os.environ.get("MAXIMUM_NUMBER_OF_EPOCHS_IDCNN", 10000)
)

IDCNN_NETWORK_NAME = "idCNN"  # "idCNN_adaptive"
THRESHOLD_EARLY_STOP_ACCUMULATION = float(
    os.environ.get("THRESHOLD_EARLY_STOP_ACCUMULATION", 0.9995)
)
THRESHOLD_ACCEPTABLE_ACCUMULATION = float(
    os.environ.get("THRESHOLD_ACCEPTABLE_ACCUMULATION", 0.9)
)
MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS = int(
    os.environ.get("MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS", 3)
)
MAXIMAL_IMAGES_PER_ANIMAL = int(
    os.environ.get("MAXIMAL_IMAGES_PER_ANIMAL", 3000)
)
RATIO_OLD = float(os.environ.get("RATIO_OLD", 0.6))
RATIO_NEW = float(os.environ.get("RATIO_NEW", 0.4))
CERTAINTY_THRESHOLD = float(os.environ.get("CERTAINTY_THRESHOLD", 0.1))
MAX_RATIO_OF_PRETRAINED_IMAGES = float(
    os.environ.get("MAX_RATIO_OF_PRETRAINED_IMAGES", 0.95)
)
MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION = float(
    os.environ.get(
        "MINIMUM_RATIO_OF_IMAGES_ACCUMULATED_GLOBALLY_TO_START_PARTIAL_ACCUMULATION",
        0.5,
    )
)
###############################################################################
# Post processing advanced parameters
###############################################################################
FIXED_IDENTITY_THRESHOLD = float(
    os.environ.get("FIXED_IDENTITY_THRESHOLD", 0.9)
)
VEL_PERCENTILE = int(os.environ.get("VEL_PERCENTILE", 99))
################################################
# After tracking advanced parameters
################################################
INDIVIDUAL_VIDEO_WIDTH_HEIGHT = None
CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON = False
