What's new in idtracker.ai v3
*****************************

- Works with Python 3.7.
- Horizontal layout for graphical user interface (GUI). This layout can be deactivated using the ``local_settings.py`` setting  ``NEW_GUI_LAYOUT=False``.
- New GUI allows to mark landmark points in the video frame that will be stored in the ``trajectories.npy`` and ``trajectories_wo_gaps.npy`` in the key ``setup_poitns``.
- Width and height of GUI can be change using the ``local_settings.py`` using the ``GUI_MINIMUM_HEIGHT`` and ``GUI_MINIMUM_WIDTH`` variables.
- Add ground truth button to validation GUI
- Neural network training works with PyTorch 1.8.1. See installation instructions.
- Crossing detector images are the same as the identification images. This safes computing time and makes the process of generating the images faster.
- Simplify code for connecting blobs from frame to frame.
- Remove unnecessary execution of the blobs connection algorithm.
- Improved code formatting using the black formatter.
- Better factorization of the TrackerApi.
- Some bugs fixed.

To check the old web page, click on the white arrow in the bottom left box.
Then click on the version 3.0.24-alpha.
This will open the documentation corresponding to the old version.
