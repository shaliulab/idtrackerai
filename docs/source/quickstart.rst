Quickstart
==========

In this section we explain how to start tracking a video with idtracker.ai. For
more information about the different functionalities of the system go to the
Graphical User Interface section :doc:`./GUI_explained`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1. Download the video example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If it is the first time that you are using this system, we recomend to start with
the video example. You can download it from
`this link <https://drive.google.com/open?id=11G4cg3lb2yvS4ppym73YO5ILwOCJlhkk>`_

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 2. Copy video to an adequate location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Copy the video to a folder where you want the output files to be placed.
Depending on the length of the video, the number of animals, and the number
of pixels per animal, idtracker.ai will generate different amounts of data,
so there must be free space on the disk to allocate the output files.

For a video of :math:`\approx18` sec at :math:`28fps` (:math:`\approx510` frames) with :math:`8` animals
and an average of :math:`334` pixels per animal the system produces :math:`227.4` MB. We recommend
using solid state disks as the saving and loading of the multiple objects that
idtracker.ai generates will be faster.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 3. Start a tracking session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
idtrackerai is equipped with a GUI. Start it by double-clicking on the
file created during installation.

If you want to start the GUI from the terminal, open it and type ::

  source activate idtrackerai-environment
  idtrackeraiGUI

.. figure:: ./_static/qs_img1.png
   :scale: 50 %
   :align: right
   :alt: welcome tab

After opening the idtracker.ai user interface, browse to the folder containing
the video you want to track (use the bar with the symbol "../" to go up in
your folders.
In case the video is split in segments, the software will concatenate them if
each segment is named according to the following convention:

    videoName_segmentNumber.extension

Click on the video file (or one of its segments) to start a new tracking session.
After clicking on the video file it is necessary to assign a name to the current
tracking session.

By entering the name of an existing session it will be possible to load all or
part of the processes already computed. Note that the word "session" is
included by default in the name, so if your session name is "newTracking" the
session folder generated will be:

    session_newTracking

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 4. Select a region of interests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If needed, it is possible to select one or more ROIs to limit the tracking to a
portion of each frame of the video or to prevent objects to be detected.

.. figure:: ./_static/qs_img3.png
   :scale: 50 %
   :align: center
   :alt: roi

First select the shape of the ROI: It is possible to draw either rectangular or
elliptic ROIs. To draw a rectangle, first click on the position of one of the
corners of the rectangle and then on the position of the opposite corner.
To generate a elliptical ROI select 5 points on the contour of the ellipse that
you want to draw.

In case of mistake it is possible to delete the ROI and draw a new one. Furthermore,
it is possible to draw as many ROIs as desired. The slider allows to browse the
video in order to check the goodness of the selected ROIs in every frame. Finally,
save the ROI by clicking the button “save ROI” before leaving the tab.

If you don't need to select a region of interest skip this tab by clicking on
the tab preprocessing.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 5. Video segmentation and preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/qs_img4.png
   :scale: 50 %
   :align: right
   :alt: preprocessing


The aim of this step is to set the parameters that allow to separate the animals
from the background.

Apply the ROIs selected in the previous step by activating the “apply ROI” switch.
The minimum and maximum threshold sliders allow to define a range of admissible
intensities. In the example, since the fish are darker than the background,
we consider only pixels whose intensity is greater or equal than :math:`135`.
The intensity ranges from :math:`0` to :math:`255`.

We call a collection of connected pixels that satisfies the intensity thresholds a blob.
The user can set the range of acceptable areas (number of pixels) of the segmented blobs.
This allows to exclude noisy blobs, or bigger objects that do not correspond
to animals, despite their intensity. The bars plot on the bottom displays the
areas of the detected objects in the current frame. A horizontal line indicates
the minimum of the areas of the segmented blobs.

.. figure:: ./_static/qs_img6.png
  :scale: 25 %
  :align: right
  :alt: welcome tab

After setting the parameters, the segmentation of the video can be initiated by
clicking the button “Segment video”. A popup showing an estimate of the number
of animals present in the video will open. Modify the number if it is incorrect
and press return on your keyboard. A series of popups will keep you updated about
the stage of the preprocessing.

idtracker.ai uses deep learning to discriminate between segmented images
representing single individuals and multiple touching animals. A final preprocessing
popup shows the graph of the loss function and the accuracy of this network,
when trained on a dataset automatically extracted from the images segmented in
the previous stages. It is now possible to start the tracking by clicking on
the bar that says "Go to the tracking tab".


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 6. Start tracking the video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/qs_img7.png
   :scale: 50 %
   :align: right
   :alt: welcome tab

To start the tracking click on the button “Start protocol cascade”. The values
displayed on the right of the tab are the hyperparameters used to initialise
the artificial neural network used to identify the animals. These parameters
can be changed by clicking on the button “Advanced idCNN controls”, we recommend
only advanced users to access this options. After clicking on the button
“Start protocol cascade” a popup will keep you updated about the state of
the algorithm.

.. figure:: ./_static/qs_img8.png
   :scale: 20 %
   :align: center
   :alt: welcome tab

After the protocol has been carried out successfully and the trajectories of
the identified animals have been saved a popup allows either to quit the program
or proceed to the validation of the video. In addition, the estimated accuracy
of the tracking is shown. The algorithm will automatically recommend the user
to proceed to the validation if the estimated accuracy is lower than expected.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 7. Global and individual validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./_static/qs_img9.png
   :scale: 50 %
   :align: right
   :alt: welcome tab

The output of the tracking algorithm can be easily validated and corrected by
using the global validation and individual validation tabs.

Since the identity of the animals is preserved between crossings, it is possible
to jump from one crossing to the next or the previous by using the “Go to next
(previous) crossing” button, or by pressing the up and down arrow on the keyboard.

The identification of the individual is done starting from a particular part of
the video called “first global fragment”. We suggest to start a validation from
this part of the video which can be reached in any moment by clicking on the
button “First global fragment”.

To modify the identity of an individual click inside of the body of the animal.
A pop up will apear indicating the current identity of the animal. Type the new
identity and press return. The new identity will be propagated to the past and
the future until the animal enters a crossing or disappears. In case the user
modifies at least one of the assigned identities the algorithm gives the possibility
to save the updated identities and updates the file were all the information about
the blobs is stored.


^^^^^^^^^^^^^^^^^^^^
Step 8. Output files
^^^^^^^^^^^^^^^^^^^^

The files generated during the tracking and the files with the trajectories
are stored in the session folder. The trajectories of the animals in the parts
where they are not crossing can be found in the folder "trajectories". The
trajectories with the interpolated position of the animals during the crossings
can be found in the folder "trajectories_wo_gaps".

.. figure:: ./_static/session_folder.png
   :scale: 80 %
   :align: center
   :alt: welcome tab
