FAQs
=====

General questions
-----------------

Can I use idtracker.ai in my videos?
************************************

You can check our :doc:`./gallery` to see the type of videos in which idtracker.ai worked well. We also give
a set of :doc:`./video_conditions` that we advise users to follow to get the best results with idtracker.ai.


Does idtracker.ai work in Windows?
**********************************

Yes, in the `Gitlab Repository <https://gitlab.com/polavieja_lab/idtrackerai>`_ we provide instructions to
install idtracker.ai in Windows. We have tested the installation in computers running Windows 10 Pro.


Can I run idtracker.ai in a laptop?
***********************************

Yes. We are running idtracker.ai with all its features in high-end performance
gaming laptops from `Obsidian <https://shop.obsidian-pc.com/en/workstation.html>`_.
If your laptop does not have a GPU you can still use idtracker.ai, see the previous FAQ.


Can I use idtracker.ai if my computer does not have a good GPU?
***************************************************************

Yes, you can still use idtracker.ai if you don't have a GPU. However, the parts of the tracking that
use the GPU will be up to 100x slower. However, if you are tracking a single animal, or if you are tracking
groups of animals but you do not care about their identities, you can use idtracker.ai and track
animals at a faster speed. To track single animals read the previous FAQ. To track animals without keeping
their identity, just click the button *Tracking without identities* when you get to the *Tracking* tab
of idtracker.ai (after the video is segmented).


Can I use idtracker.ai in an AMD GPU?
*************************************

Althought there are some efforts on making Tensorflow work with AMD GPUs, we do not
provide support or installation instructions in this case. As idtracker.ai is free and open source,
you can always download it and set your environment with Tensorflow supporting your AMD GPU. However,
we have not heard from any user using idtracker.ai with AMD GPUs, so if you are successful, please let us know!


Can I use idtracker.ai with Google Colab?
*****************************************

Currently idtracker.ai is a GUI (Graphical User Interface) based software. This means that
you cannot easily run it from a terminal. As idtracker.ai is free and open source,
you can always modify the code to run the GPU intensive parts from a terminal and hence from Google Colab.

We are currently working on a new GUI and code structure using `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_
that will be released by mid-2019. This will allow users to interact with idtracker.ai using the terminal.
Then we will provide a Jupyter Notebook with the necessary commands to run idtracker.ai in Google Colab.


Is there a Docker image for idtracker.ai?
*****************************************

Currently idtracker.ai is a GUI (Graphical User Interface) based software. Docker typically works
easily when you can run the software from the terminal (command line). Thus, for now it does not make
sense to provide a Docker image.

However, we are working on a new GUI and code structure using `Pyforms <https://pyforms.readthedocs.io/en/v4/>`_
that will be released by mid-2019. This will allow users to interact with idtracker.ai using the terminal.
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

Common installation problems
----------------------------

Some of the errors that you might encounter might have been already reported by other users and
fixed. Please update your idtracker.ai to make sure you are using the latest version. To update
idtracker.ai follow the instructions in the `idtracker.ai gitlab repository <https://gitlab.com/polavieja_lab/idtrackerai>`_

If the error persists, please report the issue in the `idtracker.ai gitlab repository <https://gitlab.com/polavieja_lab/idtrackerai>`_
or send us an email to idtrackerai@gmail.com. We will try to fix it as soon as possible.

Solving environment: failed
***************************

.. code::

    Solving environment: failed

    ResolvePackageNotFound:
    - libtiff==4.0.9=vc14_0

This error occurs when one of the libraries listed in the *environment.yml* file
(in this case the *libtiff* library) has been updated in the Conda cloud
repository and the version does not match.
You can try to solve the error by checking which is the latest version in the Conda cloud.


Common GUI (Graphical User Interface) bugs and questions
--------------------------------------------------------

We are constantly improving the GUI. However, you might still find some bugs, please report them.
The following bugs that we describe do not affect the tracking performance, and you can still use
idtracker.ai is you learn how to avoid them.

Reopening and validating a previous tracking session
****************************************************
idtracker.ai allows you to revisit previous tracking sessions to, for example, validate the trajectories of a tracked video,
or to recompute some step of the tracking pipeline.

You can do this by starting idtracker.ai and selecting again the .avi file of the video
that you want to revisit. When the GUI asks for the session name, input the name of the session you want to revisit
(exclude the *session_* part). A pop-up with the different steps computed will appear.
For example, if you restore all the steps, a pop-up will appear asking if you want to go to the
Global or Individual Validation tabs. By selecting one of them, a last pop-up will ask if you want to check the trajectories with the animals
identified during the crossings or without identities during crossings. Press one of them to validate the trajectories.

Note that you can only modify the identities in the option "With animals not identified during crossings". When you save the new modified identities
the software will automatically reinterpolated the trajectories to identify the animals during the crossings. 

Empty tabs
**********

idtracker.ai has a very lineal processing procedure. If some tabs appear empty it is because
you haven't perform the step necessary to move to the next tab. For example, at the beginning
only the Welcome tab will have content, the rest of the tabs will be empty. First you need to
select a video for the ROI Selection and Preprocessing tabs to become active. The same will occur if you
are in the preprocessing tab and try to go to the Tracking or Global Validation tabs. First
you need segment the video for the tab Tracking to become active.

ROI warning popup but ROI is selected
*************************************

When moving form the Preprocessing tab to the ROI Selection tab, and trying to save a ROI,
a ROI warning might pop with the following message:


    It seems that the ROI you are trying to apply corresponds to the entire frame.
    Please, go to the ROI Selection tab to select and save a ROI.

You should ignore it and press the save ROIs button again. Then your ROI will be saved.

idtracker.ai crashes when selecting video
*****************************************

If you select a video, the Session Name popup will appear for you to input the session name.
If you go out of this popup and try to select the video again, you might get the following error.

.. code::

    File "/home/polaviejalab/idtrackerai/idtrackerai/gui/select_file.py", line 121, in open CHOSEN_VIDEO.set_chosen_item(self.filechooser.selection[0])
    IndexError: list index out of range

Once you have selected a video, you should input the session name and pres ENTER. If you selected the
wrong video, you should close the idtracker.ai GUI and open it again.
