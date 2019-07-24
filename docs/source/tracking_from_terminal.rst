Tracking from the terminal
==========================

The new idtracker.ai v3 allows to track videos using only the command line.
This feature is particularly useful to track multiple videos sequentially
without having to interact with the GUI.

Tracking from the terminal
**************************
You can track a video using only the terminal running the following command

.. code-block:: bash

    idtrackerai terminal_mode --_video "absolute_video_folder/video.avi" --exec track_video

This will track the video with the default parameters.

To change the value of the different preprocessing parameters you just need
to add them to the same line in the following way.

.. code-block:: bash

    idtrackerai terminal_mode --_video "absolute_video_folder/video.avi" --_session session0 --_resreduct 0.3 --_intensity [0,135] --_area [5,50] --_range [0,508] --_nblobs 8 --_roi "[[(10,10),(200,10),(10,200)]]" --exec track_video


Here we put a list of the parameters name to be use in order to set the
corresponding parameter of the GUI.

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
*******************************************************

We recommend to check how the preprocessing parameters affect the detection
of the animals in the video. The previous method does not allow to do this.
However, you can set the parameters using the GUI and them save them into a
.json file using the **Save parameters** button (see the :doc:`GUI_explained`
page). Then you can track the video from the terminal using the parameters
saved in the .json file using the following command.

.. code-block::  bash

    idtrackerai terminal_mode --load absolute_path_to_json_file --exec track_video

We find this method more simple and more reliable as the user is forced to
check the preprocessing parameters in the GUI.

Tracking multiple videos sequentially using multiple .json files
****************************************************************

A good set of steps to track multiple videos sequentially could be the
following.

1. Open the idtracker.ai GUI.
2. Open a video.
3. Set the preprocessing parameters and explore that video to check that the number of blobs detected and the number of animals in the **Number of animals** box is the same for most of the video.
4. Save the parameters. We recommend saving the parameters in the same folder where the video is and with the same name as the session. This way if you start another tracking session, you can check the parameters that you used for each one.
5. Without closing the GUI, repeat 2-4 for as many videos as you want.
6. Prepare a Python script or a Shell script that executes the command for every .json file that you have created.

.. code-block:: bash

    idtrackerai terminal_mode --load absolute_path_to_json_file --exec track_video

7. Execute the Python script or the Shell script.

You can download two example Python scripts and Shell scripts for multiple video
tracking from THIS LINK.

We recommend checking how match data is generated after tracking one of the videos
and checking that you have enough space in your hardrive to save all the data
that will be generated after tracking all the videos. You can change the
amount of data stored for every tracking session changing the DATA_POLICY
advanced parameter (see :doc:`advanced_parameters`).
