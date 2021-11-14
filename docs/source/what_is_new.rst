What's new in idtracker.ai v4
*****************************

- Works with Python 3.7.
- Remove Kivy submodules and stop support for old Kivy GUI.
- Neural network training is done with Pytorch 1.10.0.
- Identification images are saved as uint 8.
- Crossing detector images are the same as the identification images. This saves computing time and makes the process of generating the images faster.
- Improve data pipeline for the crossing detector.
- Parallel saving and loading of identification images (only for Linux)
- Simplify code for connecting blobs from frame to frame.
- Remove unnecessary execution of the blobs connection algorithm.
- Background subtraction considers the ROI
- Allows to save trajectories as csv with the advanced parameter `CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON` (using the `local_settings.py` file).
- Allows to change the output width (and height) of the individual-centered videos with the advanced parameter `INDIVIDUAL_VIDEO_WIDTH_HEIGHT` (using the `local_settings.py` file).
- Horizontal layout for graphical user interface (GUI). This layout can be deactivated using the `local_settings.py` setting  `NEW_GUI_LAYOUT=False`.
- Width and height of GUI can be changed using the `local_settings.py` using the `GUI_MINIMUM_HEIGHT` and `GUI_MINIMUM_WIDTH` variables.
- Add ground truth button to validation GUI.
- Added "Add setup points" featrue to store landmark points in the video frame that will be stored in the `trajectories.npy` and `trajectories_wo_gaps.npy` in the key `setup_poitns`. Users can use this points to perform behavioural analysis that requires landmarks of the experimental setup.
- Improved code formatting using the black formatter.
- Better factorization of the TrackerApi.
- Some bugs fixed.
- Better documentation of main idtracker.ai objects (`video`, `blob`, `list_of_blobs`, `fragment`, `list_of_fragments`, `global_fragment` and `list_of_global_fragments`).
- Dropped support for MacOS