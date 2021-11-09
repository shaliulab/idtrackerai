Tracking from the terminal
==========================

The new idtracker.ai v3 allows to track videos using only the command line. This feature is particularly useful to track multiple videos sequentially without having to interact with the GUI.

Tracking from the terminal
--------------------------
You can track a video using only the terminal running the following command

.. code-block:: bash

    idtrackerai terminal_mode --_video "absolute_video_folder/video.avi" --exec track_video

This will track the video with the default parameters.

To change the values of the different preprocessing parameters you just need to add them to the same line in the following way.

.. code-block:: bash

    idtrackerai terminal_mode --_video "absolute_video_folder/video.avi" --_session session0 --_resreduct 0.3 --_intensity [0,135] --_area [5,50] --_range [0,508] --_nblobs 8 --_roi "[[(10,10),(200,10),(10,200)]]" --exec track_video


Here we put a list of the parameters name to be use in order to set the corresponding preprocessing parameter of the GUI.


+--------------------------+--------------------------------------------------+
| **Parameter name**       | **Parameter in GUI**                             |
+--------------------------+--------------------------------------------------+
| _session                 | Session                                          |
+--------------------------+--------------------------------------------------+
| _video                   | Video                                            |
+--------------------------+--------------------------------------------------+
| _video_path              | Video file. Overwrites the _video parameter      |
+--------------------------+--------------------------------------------------+
| _applyroi                | Apply                                            |
+--------------------------+--------------------------------------------------+
| _roi                     | ROI                                              |
+--------------------------+--------------------------------------------------+
| _chcksegm                | Check segmentation                               |
+--------------------------+--------------------------------------------------+
| _resreduct               | Resolution reduction                             |
+--------------------------+--------------------------------------------------+
| _intensity               | Intensity thresholds                             |
|                          |                                                  |
+--------------------------+--------------------------------------------------+
| _area                    | Area                                             |
+--------------------------+--------------------------------------------------+
| _range                   | Tracking                                         |
+--------------------------+--------------------------------------------------+
| _nblobs                  | Number of animals                                |
+--------------------------+--------------------------------------------------+
| _rangelst                | Tracking intervals                               |
+--------------------------+--------------------------------------------------+
| _multiple_range          | Multiple                                         |
+--------------------------+--------------------------------------------------+
| _no_ids                  | Track                                            |
+--------------------------+--------------------------------------------------+


Tracking from the terminal using a .json file
---------------------------------------------

We recommend to check how the preprocessing parameters affect the detection of the animals in the video. The previous method does not allow to do this. However, you can set the parameters using the GUI and them save them into a *.json* file using the **Save parameters** button (see the :doc:`GUI_explained` page). Then you can track the video from the terminal using the parameters saved in the *.json* file using the following command.

.. code-block:: bash

    idtrackerai terminal_mode --load absolute_path_to_json_file --exec track_video

We find this method more simple and more reliable as the user is forced to check the preprocessing parameters in the GUI.

If you have saved the *.json* file in a computer and you are going to track the video in a different computer, you can overwrite the path to the video in the *.json* file by running the following command:

.. code-block:: bash

    idtrackerai terminal_mode --load absolute_path_to_json_file --_video_path new_path_to_video --exec track_video

Tracking multiple videos sequentially using multiple .json files
----------------------------------------------------------------

A possible set of steps to track multiple videos sequentially could be the following.

1. Open the idtracker.ai GUI.
2. Open a video.
3. Set the preprocessing parameters and explore that video to check that the number of blobs detected and the number of animals in the **Number of animals** box is the same for most of the video.
4. Save the parameters. We recommend saving the parameters in the same folder where the video is and with the name "params.json".
5. Without closing the GUI, repeat 2-4 for as many videos as you want.
6. Prepare a Python script or a Shell script that executes the command for every *.json* file that you have created.

.. code-block:: bash

    idtrackerai terminal_mode --load absolute_path_to_json_file --exec track_video

7. Execute the Python script or the Shell script.

Example batch tracking script in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are multiple ways of writing an script in Python that scans you folders and track the videos if they contain a *.json* file. This example script assumes the following.

1. A main folder contains all the videos to be tracked.

2. Inside of the main folder the videos are placed in subfolders that contain the video and the a file named "params.json" with the preprocessing parameters that should be used to track it.

Copy the following code in a file a name it "idtrackerai_batch_tracking.py".

.. code-block:: python

    import os
    import sys

    project_directory = sys.argv[1]
    for root, subdirs, files in os.walk(project_directory):
        json_file = os.path.join(root, 'params.json')
        if os.path.isfile(json_file):
            os.system('idtrackerai terminal_mode --load {} --exec track_video'.format(json_file))

Execute the script using the following command.

.. code-block:: bash

    python idtrackerai_batch_tracking.py /path/to/mainFolder/with/all/videos

Note that you will need to substitute the '/path/to/mainFolder/with/all/videos' with the path to your folder that contains all the videos.

We recommend checking how much data is generated after tracking one of the videos and checking that you have enough space in your hard-drive to save all the data that will be generated after tracking all the videos. You can change the amount of data stored for every tracking session changing the DATA_POLICY advanced parameter (see how to do this in the :doc:`advanced_parameters` page).

Note that if you want to use a "local_settings.py" file to modify some :doc:`advanced_parameters`, this file should be in the same directory from where you execute the script "idtrackerai_batch_tracking.py".

In some situations, the user might be saving the *.json* files in a computer and tracking the videos in a different one. For that, the path of the video that is saved in the .json file needs to be overwritten. An example script for single files videos would be the following

.. code-block:: python

    import os
    import sys
    import glob

    project_directory = sys.argv[1]
    for root, subdirs, files in os.walk(project_directory):
    json_file = os.path.join(root, 'params.json')
    path_to_video = glob.glob(os.path.join(root, '*.avi'))
    if os.path.isfile(json_file) and len(path_to_video) == 1:
        path_to_video = path_to_video[0]
        os.system('idtrackerai terminal_mode --load {} --_video_path {} --exec track_video'.format(json_file, path_to_video))

Note that we have added the option --_video_path. This will overwrite the _video parameter inside of the *.json* file. 
