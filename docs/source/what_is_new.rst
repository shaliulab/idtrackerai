What's new in idtracker.ai v3
*****************************

- idtracker.ai v3 can be installed from the PyPI python package manager (see :doc:`how_to_install`)
- New Graphical User Interface (GUI) based on `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_.
- Track videos from the command line with the *terminal_mode*. Save the preprocessing parameters for a video and load them with the *terminal_mode*. This will allow you to track batches of videos sequentially without having to interact with the GUI (see :doc:`tracking_from_terminal`).
- Change advance tracking parameters using a *local_settings.py* file (see :doc:`advanced_parameters`).
- Improved memory management during tracking. Segmentation images and sets of pixels can be now saved in RAM or disk. Segmentation images and pixels are saved in the disk by default. Set the corresponding parameters using the *local_settings.py* file (see :doc:`advanced_parameters`). Saving images and pixels in the disk will make the tracking slower, but it will allow you to track longer videos with less RAM memory.
- Improved data storage management. Use the parameter *DATA_POLICY* in the *local_settings.py* file (see :doc:`advanced_parameters`) to decide which files to save after tracking. This will prevent you from storing heavy unnecessary files if what you just need the trajectories.
- Improved validation and correction of trajectories with a new GUI based on `Python Video Annotator <https://pythonvideoannotator.readthedocs.io/en/master/modules/idtrackerai.html>`_. Now you can modify the position of the centroids in individual and crossings. Also, you can use the tools from the `Python Video Annotator <https://pythonvideoannotator.readthedocs.io/en/master/modules/idtrackerai.html>`_ to annotate the behaviour in your videos.
- Overall improvements in the internal structure of the code. Algorithm and GUI are now completely separated. The idtrackerai module and API are stored in the `idtrackerai repository <https://gitlab.com/polavieja_lab/idtrackerai>`_. The new `Pyforms based GUI has its how repository <https://gitlab.com/polavieja_lab/idtrackerai-app>`_. The `validation GUI also has its own repository <https://github.com/UmSenhorQualquer/pythonvideoannotator-module-idtrackerai>`_.
- The `old Kivy-based GUI has its own repository <https://gitlab.com/polavieja_lab/idtrackerai-gui-kivy>`_. You can download it and install it from the repository. We made some changes in the code of the old GUI to make it compatible with the new idtracker.ai v3. However, the old Kivy-based GUI won't be supported in the future.
- Multiple bugs fixed.

To check the old web page, click on the white arrow in the bottom left box. Then click on the version 2.0.0-alpha. This will open the documentation corresponding to the old version.
