Welcome to idtracker.ai's documentation!
===========================================

idtracker.ai allows to track groups of up to 100 unmarked animals from videos recorded in laboratory conditions.

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

New release: idtracker.ai v3
****************************

* A more robust :doc:`GUI_explained`.
* Possibility of :doc:`tracking_from_terminal` which allow for a higher throughput pipeline.
* Modify :doc:`advanced_parameters` to optimize memory management and other features of the algorithm.

Check :doc:`what_is_new` and join the `idtracker.ai users group <https://groups.google.com/forum/#!forum/idtrackerai_users>`_ to get announcements about new releases.

Start using idtracker.ai
************************

Check the :doc:`how_to_install` to find the best installation mode for your usage case.

Follow the instructions in the :doc:`quickstart` to track the example video and get use to the idtracker.ai workflow.

If you are unsure whether idtracker.ai will work on your videos, check the :doc:`video_conditions` and the :doc:`gallery` to see how our videos look like.


Our research using idtracker.ai
*******************************
Let us know if you are using idtracker.ai in your research.

  - `Heras, F. J., Romero-Ferrero, F., Hinz, R. C., & de Polavieja, G. G. (2019). Deep attention networks reveal the rules of collective motion in zebrafish. PLoS computational biology, 15(9), e1007354. <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007354>`_
  - `Laan, A., Iglesias-Julios, M., & de Polavieja, G. G. (2018). Zebrafish aggression on the sub-second time scale: evidence for mutual motor coordination and multi-functional attack manoeuvres. Royal Society open science, 5(8), 180679. <https://royalsocietypublishing.org/doi/full/10.1098/rsos.180679#d3593705e1339>`_

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

  Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
  idtracker.ai: tracking all individuals in small or large collectives of unmarked animals `[pdf] <https://drive.google.com/open?id=1fYBcmH6PPlwy0AQcr4D0iS2Qd-r7xU9n>`_
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
   tutorials
   what_is_new
   trajectories_analysis
   gallery
   FAQs
   modules
   data
   contact
