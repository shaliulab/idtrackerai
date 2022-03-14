Welcome to idtracker.ai's documentation!
===========================================

idtracker.ai allows to track groups of up to 100 unmarked animals from videos 
recorded in laboratory conditions.

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

New release: idtracker.ai v4
********************************

* Works with Python 3.7 and Pytorch 1.10.0 and CUDA 10.2 or 11.3.
* New horizontal GUI layout.
* "Add setup points" feature allows to annotate groups of points in the frame that can be useful for analysis. These groups of points are stored together with the trajectories in the `trajectories.npy` and `trajectories_wo_gaps.npy` files.
* Save trajectories as CSV files using the advanced parameters.

Check :doc:`what_is_new` and join the 
`idtracker.ai users group <https://groups.google.com/forum/#!forum/idtrackerai_users>`_ 
to get announcements about new releases.

Start using idtracker.ai
************************

Check the :doc:`how_to_install` to find the best installation mode for your usage case.

Follow the instructions in the :doc:`quickstart` to track the example video and get use to the idtracker.ai workflow.

If you are unsure whether idtracker.ai will work on your videos, check the :doc:`video_conditions` and the :doc:`gallery` to see how our videos look like.


Our research using idtracker.ai
*******************************

  - `Heras, F. J., Romero-Ferrero, F., Hinz, R. C., & de Polavieja, G. G. (2019). Deep attention networks reveal the rules of collective motion in zebrafish. PLoS computational biology, 15(9), e1007354. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007354>`_ `[bibtex] <https://scholar.googleusercontent.com/scholar.bib?q=info:V7dp5ZkhNJ8J:scholar.google.com/&output=citation&scisdr=CgW_YpfCEPnjl2PUqPg:AAGBfm0AAAAAXYnRsPgSPNRDi8mDIRFC17q4Y3gfqJxj&scisig=AAGBfm0AAAAAXYnRsHTK4UNv5YARsNaijlcY1mjyJWwW&scisf=4&ct=citation&cd=-1&hl=en>`_ `[bioRxiv] <https://www.biorxiv.org/content/10.1101/400747v2>`_  `[gitlab] <https://gitlab.com/polavieja_lab/fishandra>`_ `[data] <https://drive.google.com/drive/folders/1Oq7JPmeY3bXqPXc_oTUwUZbHU-m4uq_5>`_.
  - `Laan, A., Iglesias-Julios, M., & de Polavieja, G. G. (2018). Zebrafish aggression on the sub-second time scale: evidence for mutual motor coordination and multi-functional attack manoeuvres. Royal Society open science, 5(8), 180679. <https://royalsocietypublishing.org/doi/full/10.1098/rsos.180679#d3593705e1339>`_ `[bibtex] <https://scholar.googleusercontent.com/scholar.bib?q=info:gmQUQmzvzucJ:scholar.google.com/&output=citation&scisdr=CgW_YpfCEPnjl2PXek0:AAGBfm0AAAAAXYnSYk0k2Z0wYPI93n58asNqyjvHMNcb&scisig=AAGBfm0AAAAAXYnSYoQQOoklpi_RE_q7-fPQ7ksOwSqm&scisf=4&ct=citation&cd=-1&hl=en>`_ `[bioRxiv] <https://www.biorxiv.org/content/10.1101/208918v2>`_


Source code
***********

The source code can be found at the `idtracker.ai Gitlab repository <https://gitlab.com/polavieja_lab/idtrackerai>`_.

Check the code documentation :ref:`genindex` for more information about different classes, functions and methods of idtracker.ai

Data
****

The data used in the idtracker.ai article can be found in the :doc:`./data` section of this web page.

Reference
*********
When using information from this web page please reference

  `Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
  idtracker.ai: tracking all individuals in small or large collectives of unmarked animals <https://www.nature.com/articles/s41592-018-0295-5>`_ `[pdf] <https://drive.google.com/open?id=1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n>`_ `[bibtex] <https://scholar.googleusercontent.com/scholar.bib?q=info:9t2LqPxDOpUJ:scholar.google.com/&output=citation&scisdr=CgW_YpfCEPnjl2PUEbE:AAGBfm0AAAAAXYnRCbEqxXF_BhL0yAml4NwFJQvgEVTl&scisig=AAGBfm0AAAAAXYnRCXv6FF-rvpeJlUvW6JVTgZgqwmI7&scisf=4&ct=citation&cd=-1&hl=en>`_ `[arXiv] <https://arxiv.org/abs/1803.04351>`_.
  (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

Search in this webpage
**********************

* :ref:`search`
.. * :ref:`modindex`

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
   setups
   tutorials
   what_is_new
   trajectories_analysis
   gallery
   FAQs
   modules
   data
   contact
