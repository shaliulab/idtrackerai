FAQs
=====

Can I use idtracker.ai in my videos?
************************************

You can check our :doc:`./gallery` to see the type of videos in which idtracker.ai worked well. We also give
a set of :doc:`./video_conditions` that we advise users to follow to get the best results with idtracker.ai.


Does idtracker.ai work in Windows?
**********************************

Yes, in the `Gitlab Repository <https://gitlab.com/polavieja_lab/idtrackerai>`_ we provide instructions to
install idtracker.ai in Windows. We have tested the intallation in computers running Windows 10 Pro.


Can I run idtracker.ai in a laptop?
***********************************

Yes. We are running idtracker.ai with all its possibilities in high-end performance
gamming laptops from `Obsidian <https://shop.obsidian-pc.com/en/workstation.html>`_.
If your laptop does not have a GPU you can still use idtracker.ai, see the previous FAQ.


Can I use idtracker.ai if my computer does not have a good GPU?
***************************************************************

Yes, you can still use idtracker.ai if you don't have a GPU. However, the parts of the tracking that
use the GPU will be up to 100x slower. However, if you are tracking a single animal, or you are tracking
groups of animals but you do not care about their identities, you can use idtracker.ai and track
animals at a faster speed. To track single animals read the previous FAQ. To track animals without keeping
their identity, just click the button *Tracking without identities* when you get to the *Tracking* tab
of idtracker.ai (after the video is segmented).


Can I use idtracker.ai in an AMD GPU?
*************************************

Althought there are some efforts on making Tensorflow work with AMD GPUs, we do not
provide support or installation instructions in this case. As idtracker.ai is free and open source,
you can always download it and set your environment with Tensorflow supporting your AMD GPU. However,
we are not aware of any user using idtracker.ai with AMD GPUs.


Can I use idtracker.ai with Google Colab?
*****************************************

Currently idtracker.ai is a GUI (Graphical User Interface) based software. This means that
you cannot easily run it from a terminal. As idtracker.ai is free and open source,
you can always modify and twick the code to run the GPU intensive parts from a terminal and hence from Google Colab.

We are working on a new GUI and code structure using `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_
that will be released very soon. This will allow to interact with idtracker.ai using the terminal.
Then we will provide a Jupyter Notebook with the necessary commands to run idtracker.ai in Google Colab.


Do we provide a Docker image for idtracker.ai?
**********************************************

Currently idtracker.ai is a GUI (Graphical User Interface) based software. Docker typically works
easily when you can run the software from the terminal (command line). Thus, for now it does not make
sense to provide a Docker image.

However, we are working on a new GUI and code structure using `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_
that will be released very soon. This will allow to interact with idtracker.ai using the terminal.
Then we will provide a Docker image that you will be able to use to run idtracker.ai from the terminal.


Can idtracker.ai track multiple videos in batch?
************************************************

Currently to track every video you need to do it from the idtracker.ai GUI. In the near future we will
release a version of idtracker.ai where you will be able to set the preprocessing parameters for
multiple videos and then use a bash script to tell idtracker.ai to track those video, either in your computer
or in a cluster.


Does idtracker.ai give orientation and posture information?
***********************************************************

Orientation and posture can be computed a posteriori from the *blobs_collection.npy* file
that idtracker.ai generates. In `this repository <https://gitlab.com/polavieja_lab/midline>`_
we provide an example where we compute the nose, tail and midline for fish.

You can also `generate a small video for every animal <https://gitlab.com/polavieja_lab/idtrackerai_notebooks/blob/master/idtrackerai_helpers/extract_single_animal_video.ipynb>`_
in the group and use it to get the posture with one of the AI based posture trackings
(`Deeplabcut <https://github.com/AlexEMG/DeepLabCut>`_, `LEAP <https://github.com/talmo/leap>`_, ...).


Can I use idtracker.ai together with DeepLabCut or LEAP?
********************************************************

As explained before, you can `generate a small video for every animal <https://gitlab.com/polavieja_lab/idtrackerai_notebooks/blob/master/idtrackerai_helpers/extract_single_animal_video.ipynb>`_
in the group and use it to track the body parts with one of the AI based posture trackings
(`Deeplabcut <https://github.com/AlexEMG/DeepLabCut>`_, `LEAP <https://github.com/talmo/leap>`_, ...).


Does idtracker.ai track single animals?
***************************************

Yes. Although idtracker.ai is designed to track multiple animals keeping their
identities along the video, you can also track videos with a single animal. When
the system asks for the number of animals, just type :math:`1` and follow the
instructions. The system will automatically skip the GPU intensive parts that are
not necessary in this case. This means that you can use idtracker.ai to track
single animals in a desktop or a laptop computer even if it does not have
a GPU. In this case, idtracker.ai will perform very fast as it will only need to
segment the video to extract the position of yoru animal.


Can I track humans with idtracker.ai?
*************************************

We haven't tried to track people with idtracker.ai. We think that idtracker.ai can track well
people in videos recorded under laboratory conditions. Tracking humans on natural environments
(streets, parks,...) it is a much more difficult task for which idtracker.ai was not designed.
However, as idtracker.ai is free and open source, you can maybe use parts of our algorithm
to set your human tracking for natural environments.
