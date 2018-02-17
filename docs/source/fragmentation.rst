Fragmentation
=============


^^^^^^^^
Fragment
^^^^^^^^

A fragment is a collection of Blob objects (see :class:`~blob..Blob`) that refer to the same individual. A Fragment object manages such a collection of Blob objects
to facilitate the fingerprint operations and later on the identification of the animal reprpresented in the fragment.

.. automodule:: fragment
  :members:

^^^^^^^^^^^^^^^^^
List of fragments
^^^^^^^^^^^^^^^^^

Collection of instances of the class :class:`~fragment.Fragment`

.. automodule:: list_of_fragments
   :members:

^^^^^^^^^^^^^^^
Global fragment
^^^^^^^^^^^^^^^

Global fragments are the core of the tracking algorithm: They are collection of instances of the class :class:`~fragment.Fragment`
that contains images extracted from a part of the video in which all the animals are visible.

.. automodule:: globalfragment
  :members:

^^^^^^^^^^^^^^^^^^^^^^^^
List of global fragments
^^^^^^^^^^^^^^^^^^^^^^^^

Collection of instances of the class :class:`~globalfragment.GlobalFragment`.
Global fragments are used to create the dataset of labelled images used to train
the idCNN during the fingerprint protocols cascade.

.. automodule:: list_of_global_fragments
   :members:
