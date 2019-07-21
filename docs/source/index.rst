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
===============================

- We made the installation easier by putting idtracker.ai in the PyPI package manager.
- New Graphical User Interface (GUI) based on [Pyforms](https://pyforms.readthedocs.io/en/v4/).
- Track videos from the command line with the *terminal_mode*.
- Save the preprocessing parameters for a video and load them with the *terminal_mode*. This will allow you to track batches of videos sequentially without having to interact with the GUI.
- Change advance tracking parameters using a *local_settings.py* file.
- Improved memory management during tracking. Segmentation images and sets of pixels can be now saved in RAM or DISK. Identification images are saved in DISK. Set these parameters using the *local_settings.py* file. Saving images and pixels in the DISK will make the tracking slower, but will allow you to track longer videos with less RAM memory.
- Improved data storage management. Use the parameter *DATA_POLICY* in the *local_settings.py* file to decide which files to save after the tracking. For example, this will prevent you from storing heavy unnecessary files if what you only need are the trajectories.
- Improved validation and correction of trajectories with a new GUI based on [Python Video Annotator](https://pythonvideoannotator.readthedocs.io/en/master/).
- Overall improvements in the internal structure of the code.
- Multiple bugs fixed.

`Installation and git repository <https://gitlab.com/polavieja_lab/idtrackerai>`_
*******************************
The source code and installation instructions can be found at https://gitlab.com/polavieja_lab/idtrackerai.git.

:doc:`Quickstart <./quickstart>`
********************************
Check out the :doc:`./quickstart` to learn how to use the software.

:doc:`Video conditions <./video_conditions>`
********************************************
Check out the :doc:`./video_conditions` to get some advice in how to create videos
that will give you the best tracking results with idtracker.ai.

:doc:`Gallery <./gallery>`
**************************
Send us your videos using `idtracker.ai <http://idtracker.ai/>`_ and we will add them to our gallery with proper attribution.

:doc:`Frequently Asked Questions (FAQs) <./FAQs>`
*************************************************
We have summarized in the :doc:`./FAQs` some of the questions that we get more often.

:doc:`Trajectory analysis <./trajectories_analysis>`
****************************************************
We provide a set of :doc:`./trajectories_analysis` of the trajectories that idtracker.ai outputs.

:doc:`Code documentation <./modules>`
*************************************
`idtracker.ai <http://idtracker.ai/>`_ is opensource and free software (both as in freedom and as in free beer).
We have documented the code so that it is easier for developers to modify it to their needs.

`idmatcher.ai <https://gitlab.com/polavieja_lab/idmatcherai>`_ matches identities between videos
************************************************************************************************
We have also developed a toolbox to match identities between videos, called `idmatcher.ai <https://gitlab.com/polavieja_lab/idmatcherai>`_.

Research using idtracker.ai
***************************
Let us know if you are using `idtracker.ai <http://idtracker.ai/>`_ in your research.

  - `Heras, F. J., Romero-Ferrero, F., Hinz, R. C., & de Polavieja, G. G. (2018). Deep attention networks reveal the rules of collective motion in zebrafish. bioRxiv, 400747. <https://www.biorxiv.org/content/early/2018/12/21/400747>`_
  - `Laan, A., Iglesias-Julios, M., & de Polavieja, G. G. (2018). Zebrafish aggression on the sub-second time scale: evidence for mutual motor coordination and multi-functional attack manoeuvres. Royal Society open science, 5(8), 180679. <https://royalsocietypublishing.org/doi/full/10.1098/rsos.180679#d3593705e1339>`_


References
**********
When using information from this web page please reference

  `Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
  idtracker.ai: tracking all individuals in small or large collectives of unmarked animals <https://rdcu.be/bgN2R>`_
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


Contents
********

.. toctree::
   :maxdepth: 1

   video_conditions
   quickstart
   GUI_explained
   tracking_from_terminal
   advanced_parameters
   trajectories_analysis
   gallery
   FAQs
   modules
   data


Documentation index and search
******************************

* :ref:`genindex`
* :ref:`search`
.. * :ref:`modindex`
