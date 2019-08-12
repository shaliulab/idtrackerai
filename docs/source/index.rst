Welcome to idtracker.ai's v3 documentation!
===========================================

`idtracker.ai <http://idtracker.ai/>`_ allows to track groups of up to 100 unmarked animals from videos.

.. image:: /_static/fish_tracked.png
    :width: 150
    :target: https://www.youtube.com/watch?v=Imz3xvPsaEw
.. image:: /_static/flies_tracked.png
    :width: 150
    :target: https://www.youtube.com/watch?v=_M9xl4jBzVQ
.. image:: /_static/14ants.png
    :width: 150
    :target: https://www.youtube.com/watch?v=d0TTdu41NoA
.. image:: /_static/mice.png
    :width: 150
    :target: https://www.youtube.com/watch?v=ANsThSPgBFM
.. image:: /_static/2fish.png
    :width: 150
    :target: https://www.youtube.com/watch?v=dT28-VcXaCc

What is new in idtracker.ai v3?
*******************************

- idtracker.ai v3 can be installed from the PyPI python package manager (see :doc:`how_to_install`)
- New Graphical User Interface (GUI) based on `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_.
- Track videos from the command line with the *terminal_mode*. Save the preprocessing parameters for a video and load them with the *terminal_mode*. This will allow you to track batches of videos sequentially without having to interact with the GUI. (see :doc:`tracking_from_terminal`)
- Change advance tracking parameters using a *local_settings.py* file (see :doc:`advanced_parameters`).
- Improved memory management during tracking. Segmentation images and sets of pixels can be now saved in RAM or DISK. Segmentation images and pixels are saved in the disk by default. Set these parameters using the *local_settings.py* file. Saving images and pixels in the diks will make the tracking slower, but it will allow you to track longer videos with less RAM memory.
- Improved data storage management. Use the parameter *DATA_POLICY* in the *local_settings.py* file to decide which files to save after the tracking. For example, this will prevent you from storing heavy unnecessary files if what you only need are the trajectories.
- Improved validation and correction of trajectories with a new GUI based on `Python Video Annotator <https://pythonvideoannotator.readthedocs.io/en/master/modules/idtrackerai.html>`_.
- Overall improvements in the internal structure of the code. Algorithm and GUI are now completely separated. The idtrackerai module and API are stored in the `idtrackerai repository <https://gitlab.com/polavieja_lab/idtrackerai>`_. The new `Pyforms based GUI has its how repository <https://gitlab.com/polavieja_lab/idtrackerai-app>`_.
- The `old Kivy-based GUI has its own repository <https://gitlab.com/polavieja_lab/idtrackerai-gui-kivy>`_. You can download it and install it from the repository. We made some changes in the code of the old GUI to make it compatible with the new idtracker.ai v3. However, the Kivy-based GUI won't be supported in the future.
- Multiple bugs fixed.

Contents
********

.. toctree::
   :maxdepth: 1

   how_to_install
   video_conditions
   quickstart
   GUI_explained
   validation_GUI_explained
   tracking_from_terminal
   advanced_parameters
   tutorials
   trajectories_analysis
   gallery
   FAQs
   modules
   data

Research using idtracker.ai
***************************
Let us know if you are using `idtracker.ai <http://idtracker.ai/>`_ in your research.

  - `Heras, F. J., Romero-Ferrero, F., Hinz, R. C., & de Polavieja, G. G. (2018). Deep attention networks reveal the rules of collective motion in zebrafish. bioRxiv, 400747. <https://www.biorxiv.org/content/early/2018/12/21/400747>`_
  - `Laan, A., Iglesias-Julios, M., & de Polavieja, G. G. (2018). Zebrafish aggression on the sub-second time scale: evidence for mutual motor coordination and multi-functional attack manoeuvres. Royal Society open science, 5(8), 180679. <https://royalsocietypublishing.org/doi/full/10.1098/rsos.180679#d3593705e1339>`_


References
**********
When using information from this web page please reference

  `Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
  idtracker.ai: tracking all individuals in small or large collectives of unmarked animals <https://drive.google.com/open?id=1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n>`_
  (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

.. code-block:: bibtex

  @article{romero2019idtracker,
  title={idtracker.ai: tracking all individuals in small or large collectives of unmarked animals},
  author={Romero-Ferrero, Francisco and Bergomi, Mattia G and Hinz, Robert C and Heras, Francisco JH and de Polavieja, Gonzalo G},
  journal={Nature methods},
  volume={16},
  number={2},
  pages={179},
  year={2019},
  publisher={Nature Publishing Group}
  }


Find `here the preprint <https://arxiv.org/abs/1803.04351>`_ version of the manuscript.

Data
****
The data used in the article can be found in the :doc:`./data` section of this webpage.


Documentation index and search
******************************

* :ref:`genindex`
* :ref:`search`
.. * :ref:`modindex`
