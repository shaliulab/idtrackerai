DCD: Deep Crossing detector
===========================

Given a collection of Blob objects (see :class:`~blob..Blob`), the crossing detector module allows
to apply a pre-computed model of the area to each of the blobs and, if specified,
use a function approximator (in this case a convolutional neural network)
in order to distinguish between Blob representing single individual and touching individuals.

.. automodule:: crossing_detector
  :members:
