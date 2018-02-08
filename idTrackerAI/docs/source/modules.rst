Modules
=======

^^^^^^^^^^^^^^^^^^^^
Accumulation manager
^^^^^^^^^^^^^^^^^^^^

.. automodule:: accumulation_manager
   :members:

^^^^^^^^^^^
Accumulator
^^^^^^^^^^^

The accumulator module contains the main routine used to compute the accumulation process, which
is an essential part of both the second and third fingerprint protocol.

.. automodule:: accumulator
   :members:

^^^^^^^^
Assigner
^^^^^^^^

.. automodule:: assigner
   :members:

^^^^^
Blob
^^^^^

.. automodule:: blob
  :members:


^^^^^^^^^^^^^^^^^
Crossing detector
^^^^^^^^^^^^^^^^^

Given a collection of Blob objects (see :class:`.Blob`), the crossing detector module allows to apply a pre-computed model of the area to each of the blobs and, if specified,
use a function approximator (in this case a convolutional neural network) in order to distinguish between Blob representing single individual and touching individuals.

.. automodule:: crossing_detector
  :members:

^^^^^^^^
Fragment
^^^^^^^^

A fragment is a collection of Blob objects (see :class:`.Blob`) that refer to the same individual. A Fragment object manages such a collection of Blob objects
to facilitate the fingerprint operations and later on the identification of the animal reprpresented in the fragment.

.. automodule:: fragment
  :members:

^^^^^^^^^^^^
Segmentation
^^^^^^^^^^^^

.. automodule:: segmentation
  :members:

^^^^^^^^^^^^^^^^^^
Segmentation utils
^^^^^^^^^^^^^^^^^^

.. automodule:: video_utils
  :members:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Impossible velocity jumps correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: correct_impossible_velocity_jumps
  :members:
