from __future__ import absolute_import, division, print_function
import os
import sys
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.network_params")

class NetworkParams(object):
    """Manages the network hyperparameters and other variables related to the
    identification model (see :class:`~idCNN`)

    Attributes
    ----------
    video_path : string
        Path to the video file
    number_of_animals : int
        Number of animals in the video
    learning_rate : float
        Learning rate for the optimizer
    keep_prob : float
        Dropout probability
    _restore_folder : string
        Path to the folder where the model to be restored is
    _save_folder : string
        Path to the folder where the checkpoints of the current model are stored
    _knowledge_transfer_folder : string
        Path to the folder where the model to be used for knowledge transfer is saved
    use_adam_optimiser : bool
        Flag indicating to use the Adam optimizer with the parameters indicated in _[2]
    scopes_layers_to_optimize : list
        List with the scope names of the layers to be optimized
    _cnn_model : int
        Number indicating the model number to be used from the dictionary of models
        CNN_MODELS_DICT in :mod:`id_CNN`
    image_size : tuple
        Tuple (height, width, channels) for the input images
    number_of_channels : int
        Number of channels of the input image

    .. [2] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
    """
    def __init__(self, number_of_animals, cnn_model = 0, learning_rate = None,
                keep_prob = None, use_adam_optimiser = False,
                scopes_layers_to_optimize = None, restore_folder = None,
                save_folder = None, knowledge_transfer_folder = None,
                image_size = None,
                number_of_channels = None,
                video_path = None):

        self.video_path = video_path
        self.number_of_animals = number_of_animals
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self._restore_folder = restore_folder
        self._save_folder = save_folder
        self._knowledge_transfer_folder = knowledge_transfer_folder
        self.use_adam_optimiser = use_adam_optimiser
        self.scopes_layers_to_optimize = scopes_layers_to_optimize
        self._cnn_model = cnn_model
        self.image_size = image_size
        self.target_image_size = None
        self.pre_target_image_size = None
        self.action_on_image = None
        self.number_of_channels = number_of_channels

    def __init__(self, number_of_animals, cnn_model = 0, learning_rate = None,
                keep_prob = None, use_adam_optimiser = False,
                scopes_layers_to_optimize = None, restore_folder = None,
                save_folder = None, knowledge_transfer_folder = None,
                image_size = None,
                number_of_channels = None,
                video_path = None):

        self.video_path = video_path
        self.number_of_animals = number_of_animals
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self._restore_folder = restore_folder
        self._save_folder = save_folder
        self._knowledge_transfer_folder = knowledge_transfer_folder
        self.use_adam_optimiser = use_adam_optimiser
        self.scopes_layers_to_optimize = scopes_layers_to_optimize
        self._cnn_model = cnn_model
        self.image_size = image_size
        self.target_image_size = None
        self.pre_target_image_size = None
        self.action_on_image = None
        self.number_of_channels = number_of_channels

    @property
    def cnn_model(self):
        return self._cnn_model

    @property
    def restore_folder(self):
        return self._restore_folder

    @restore_folder.setter
    def restore_folder(self, path):
        assert os.path.isdir(path)
        self._restore_folder = path

    @property
    def save_folder(self):
        return self._save_folder

    @save_folder.setter
    def save_folder(self, path):
        if not os.path.isdir(path):
            os.path.makedirs(path)
        self._save_folder = path

    @property
    def knowledge_transfer_folder(self):
        return self._knowledge_transfer_folder

    @knowledge_transfer_folder.setter
    def knowledge_transfer_folder(self, path):
        assert os.path.isdir(path)
        self._knowledge_transfer_folder = path
