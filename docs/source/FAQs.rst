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

Yes, in the :doc:`host_to_install`_ we provide instructions to install idtracker.ai in Windows. We have tested the installation in computers running Windows 10 Pro.


Can I run idtracker.ai in a laptop?
***********************************

Yes. We are running idtracker.ai with all its features in high-end performance gaming laptops from `Obsidian <https://shop.obsidian-pc.com/en/workstation.html>`_. If your laptop does not have a GPU you can still use idtracker.ai, see the next FAQ.


Can I use idtracker.ai if my computer does not have a good GPU?
***************************************************************

Yes, you can still use idtracker.ai if you don't have a GPU. However, the parts of the tracking that use the GPU will be up to 100x slower. However, if you are tracking a single animal, or if you are tracking groups of animals but you do not want to keep their identities, you can use idtracker.ai and track animals at a faster speed.

To track animals without keeping their identity, just check the box *Track without identities* in the bottom left corner of the GUI (see :doc:`GUI_explained`).

Can I use idtracker.ai in an AMD GPU?
*************************************

Althought there are some efforts on making Tensorflow work with AMD GPUs, we do not provide support or installation instructions in this case. As idtracker.ai is free and open source, you can always download it and set your environment with Tensorflow supporting your AMD GPU. However, we have not heard from any user using idtracker.ai with AMD GPUs, so if you are successful, please let us know!

Can I use idtracker.ai with Google Colab?
*****************************************

*coming soon*


Is there a Docker image for idtracker.ai?
*****************************************

*coming soon*


Can idtracker.ai track multiple videos in batch?
************************************************

Yes, check the :doc:`tracking_from_terminal`.

Does idtracker.ai give orientation and posture information?
***********************************************************

Orientation and posture can be computed a posteriori from the *blobs_collection.npy* file
that idtracker.ai generates. In `this repository <https://gitlab.com/polavieja_lab/midline>`_
we provide an example where we compute the nose, tail and midline for fish.

You can also `generate a small video for every animal <https://gitlab.com/polavieja_lab/idtrackerai_notebooks/blob/master/idtrackerai_helpers/extract_single_animal_video.ipynb>`_ in the group and use it to get the posture with one of the AI based posture trackings (`Deeplabcut <https://github.com/AlexEMG/DeepLabCut>`_, `LEAP <https://github.com/talmo/leap>`_, ...).

Can I use idtracker.ai together with DeepLabCut or LEAP?
********************************************************

As explained before, you can `generate a small video for every animal <https://gitlab.com/polavieja_lab/idtrackerai_notebooks/blob/master/idtrackerai_helpers/extract_single_animal_video.ipynb>`_ in the group and use it to track the body parts with one of the AI based posture trackings (`Deeplabcut <https://github.com/AlexEMG/DeepLabCut>`_, `LEAP <https://github.com/talmo/leap>`_, ...).

Does idtracker.ai track single animals?
***************************************

Yes. Although idtracker.ai is designed to track multiple animals keeping their identities along the video, you can also track videos with a single animal. Just indicate that the number of animals to track is 1 in the corresponding text box in the GUI. The system will automatically skip the GPU intensive parts that are not necessary in this case. This means that you can use idtracker.ai to track single animals in a desktop or a laptop computer even if it does not have a GPU. In this case, idtracker.ai will run faster as it will only need to segment the video to extract the position of the animal.

Can I track humans with idtracker.ai?
*************************************

We haven't tried to track people with idtracker.ai. We think that idtracker.ai can track well people in videos recorded under laboratory conditions. Tracking humans on natural environments (streets, parks,...) it is a much more difficult task for which idtracker.ai was not designed. However, as idtracker.ai is free and open source, you can maybe use parts of our algorithm to set your human tracking for natural environments.

Common installation problems
----------------------------

Some of the errors that you might encounter might have been already reported by other users and fixed. Please update your idtracker.ai to make sure you are using the latest version. To update idtracker.ai follow the indtructions at the end of the :doc:`how_to_install`_ page.

If the error persists, please report the issue in the `idtracker.ai gitlab repository <https://gitlab.com/polavieja_lab/idtrackerai>`_ or send us an email to idtrackerai@gmail.com. We will try to fix it as soon as possible.

Common GUI (Graphical User Interface) bugs and questions
--------------------------------------------------------

*comming soon*
