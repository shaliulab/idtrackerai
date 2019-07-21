Setting advanced parameters
===========================

The new idtracker.ai v3 allows to set a number of advanced parameters that will
allow the user to modify how idtracker.ai behaves in different parts of the
algorithm (e.g. memory management, data storage, knowledge transfer...).

To set this advanced parameters, you will need to create a file called
"local_settings.py". You can use your favorite text editor to do so. Just
remember to save it with the extension ".py".

This file needs to be placed in the same folder where you execute the command
"idtrackerai". For example, in Linux, if you execute "idtrackerai" in the default
directory (indicated by a ~ symbol) when opening a terminal

.. figure:: ./_static/advanced_parameters/running_from_home.png
   :scale: 100 %
   :align: center
   :alt: running from home

The file will need to be stored in the directory "/home/polaviejalab" where
you will need to change "polaviejalab" by your username.

If you save the file "local_settings.py" in some folder in your desktop. You
will need to use the command "cd" the move to that directory.

.. figure:: ./_static/advanced_parameters/running_from_other_folder.png
   :scale: 100 %
   :align: center
   :alt: running from other folder

You can make sure that the "local_settings.py" file is in the directory where
you are placed by running the command "ls".

.. figure:: ./_static/advanced_parameters/check_local_settings.png
   :scale: 100 %
   :align: center
   :alt: check local settings

Let's talk now about the content of the "local_settings.py" file.

This file must contain the assignment of the new values to each advanced
parameter. For example, if we want to modify the default number of animals in
the GUI, so that we do not need to type every time the number of animals in
your experiment, you can include in the "local_settings.py" file the following
line:

    NUMBER_OF_ANIMALS_DEFAULT=5

Add this line to the "local_settings.py" file, save it and launch the
idtrackerai GUI to see the effect.



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
"""***MAX_AREA_LOWER, MAX_AREA_UPPER***
Lower and upper bounds for the maximum area slider
"""
MAX_AREA_LOWER, MAX_AREA_UPPER = 0, 60000
"""***MIN_THRESHOLD, MAX_AREA_UPPER***
Lower and upper bounds for the maximum area slider
"""
MIN_THRESHOLD, MAX_THRESHOLD = 0, 255
"""***MIN_THRESHOLD_DEFAULT, MAX_THRESHOLD_DEFAULT***
Default value for min area in preprocessing
"""
MIN_THRESHOLD_DEFAULT, MAX_THRESHOLD_DEFAULT = 0, 155




"""***NUMBER_OF_CORES_FOR_BACKGROUND_SUBTRACTION***
Number of jobs used to compute the background model
"""
NUMBER_OF_CORES_FOR_BACKGROUND_SUBTRACTION = None # Set None to use the default mode of the system. (see segmentation_utils.py module for details)
"""***NUMBER_OF_CORES_FOR_SEGMENTATION***
Number of jobs used to perform the segmentation
"""
NUMBER_OF_CORES_FOR_SEGMENTATION = None # Set None to use the default mode of the system. (see segmentation.py module for details)




"""***SAVE_PIXELS***
Where to store the pixels list of every blob
DISK: pixels are stored in hdf5 files in the disk
RAM: pixels are stored in the blob object in the ram
NOT: pixels are not stored and are computed from the contour everytime
"""
SAVE_PIXELS = 'DISK' # 'RAM' or 'NOT'
"""***SAVE_SEGMENTATION_IMAGE***
Where to store the segmentation image of every blob
DISK: image is stored in hdf5 files in the disk
RAM: image is stored in the blob object in the ram
NOT: image is not stored and are computed from the contour everytime
"""
SAVE_SEGMENTATION_IMAGE = 'DISK' # 'RAM' or 'NOT'
"""***PLOT_CROSSING_DETECTOR***
"""
PLOT_CROSSING_DETECTOR=False
'''***PLOT_ACCUMULATION_STEPS***
'''
PLOT_ACCUMULATION_STEPS = False



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
