# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys

from confapp import conf

import logging
logger = logging.getLogger("__main__.network_params")

# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import numpy as np
from confapp import conf

import logging

logger = logging.getLogger("__main__.network_params_identification")


class NetworkParams(object):
    def __init__(self,
                 number_of_classes,
                 architecture=None,
                 use_adam_optimiser=False,
                 restore_folder='',
                 save_folder='',
                 knowledge_transfer_model_file=None,
                 scopes_layers_to_optimize=None,
                 image_size=None,
                 loss='CE',
                 print_freq=2,
                 use_gpu=True,
                 optimizer='SGD',
                 schedule=None,
                 optim_args=None,
                 apply_mask=False,
                 dataset=None,
                 skip_eval=False,
                 epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
                 plot_flag=True,
                 return_store_objects=False,
                 saveid='',
                 model_name='',
                 model_file='',
                 layers_to_optimize=None,
                 video_path=None
                 ):

        self.number_of_classes = number_of_classes
        self.architecture = architecture
        self._restore_folder = restore_folder
        self._save_folder = save_folder
        self._knowledge_transfer_model_file = knowledge_transfer_model_file
        self.use_adam_optimiser = use_adam_optimiser
        self.image_size = image_size
        self.loss = loss
        self.use_gpu = use_gpu
        self.print_freq = print_freq
        self.optimizer = optimizer
        self.schedule = schedule
        self.optim_args = optim_args
        self.apply_mask = apply_mask
        self.dataset = dataset
        self.skip_eval = skip_eval
        self.epochs = epochs
        self.plot_flag = plot_flag
        self.return_store_objects = return_store_objects
        self.saveid = saveid
        self.model_name = model_name
        self.layers_to_optimize = layers_to_optimize,
        self.video_path = video_path
        self.scopes_layers_to_optimize = scopes_layers_to_optimize
        self.model_file = model_file

        if self.optimizer == 'SGD' and self.optim_args is not None:
            self.optim_args['momentum'] = 0.9


    @property
    def load_model_path(self):
        return os.path.join(self.restore_folder, self.model_file_name + '.model.pth')

    @property
    def save_model_path(self):
        return os.path.join(self.save_folder, self.model_file_name)

    @property
    def model_file_name(self):
        return "%s_%s_%s" % (self.dataset, self.model_name, self.saveid)

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
    def knowledge_transfer_model_file(self):
        return self._knowledge_transfer_model_file

    @knowledge_transfer_model_file.setter
    def knowledge_transfer_model_file(self, path):
        assert os.path.isdir(path)
        self._knowledge_transfer_model_file = path

    def save(self):
        np.save(os.path.join(self.save_folder, 'model_params.npy'), self.__dict__)


# class NetworkParams(object):
#     """Manages the network hyperparameters and other variables related to the
#     identification model (see :class:`~idCNN`)
#
#     Attributes
#     ----------
#     video_path : string
#         Path to the video file
#     number_of_animals : int
#         Number of animals in the video
#     learning_rate : float
#         Learning rate for the optimizer
#     keep_prob : float
#         Dropout probability
#     _restore_folder : string
#         Path to the folder where the model to be restored is
#     _save_folder : string
#         Path to the folder where the checkpoints of the current model are stored
#     _knowledge_transfer_folder : string
#         Path to the folder where the model to be used for knowledge transfer is saved
#     use_adam_optimiser : bool
#         Flag indicating to use the Adam optimizer with the parameters indicated in _[2]
#     scopes_layers_to_optimize : list
#         List with the scope names of the layers to be optimized
#     _cnn_model : int
#         Number indicating the model number to be used from the dictionary of models
#         CNN_MODELS_DICT in :mod:`id_CNN`
#     image_size : tuple
#         Tuple (height, width, channels) for the input images
#     number_of_channels : int
#         Number of channels of the input image
#     kt_conv_layers_to_discard : str
#         convolutional layers to discard when performing knowledge transfer
#
#     .. [2] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
#     """
#
#     def __init__(self, number_of_animals, cnn_model = conf.CNN_MODEL, learning_rate = None,
#                 keep_prob = None,
#                 use_adam_optimiser = conf.USE_ADAM_OPTIMISER,
#                 scopes_layers_to_optimize = None, restore_folder = None,
#                 save_folder = None, knowledge_transfer_folder = conf.KNOWLEDGE_TRANSFER_FOLDER_IDCNN,
#                 image_size = None,
#                 number_of_channels = None,
#                 video_path = None):
#         self.video_path = video_path
#         self.number_of_animals = number_of_animals
#         self.learning_rate = learning_rate
#         self.keep_prob = keep_prob
#         self._restore_folder = restore_folder
#         self._save_folder = save_folder
#         self._knowledge_transfer_folder = knowledge_transfer_folder
#         self.use_adam_optimiser = use_adam_optimiser
#         self.scopes_layers_to_optimize = scopes_layers_to_optimize
#         self._cnn_model = cnn_model
#         self.image_size = image_size
#         self.target_image_size = None
#         self.pre_target_image_size = None
#         self.action_on_image = None
#         self.number_of_channels = number_of_channels
#         self._kt_conv_layers_to_discard = None
#
#     @property
#     def cnn_model(self):
#         return self._cnn_model
#
#     @property
#     def restore_folder(self):
#         return self._restore_folder
#
#     @restore_folder.setter
#     def restore_folder(self, path):
#         assert os.path.isdir(path)
#         self._restore_folder = path
#
#     @property
#     def save_folder(self):
#         return self._save_folder
#
#     @save_folder.setter
#     def save_folder(self, path):
#         if not os.path.isdir(path):
#             os.path.makedirs(path)
#         self._save_folder = path
#
#     @property
#     def knowledge_transfer_folder(self):
#         return self._knowledge_transfer_folder
#
#     @property
#     def kt_conv_layers_to_discard(self):
#         return self._kt_conv_layers_to_discard
#
#     @knowledge_transfer_folder.setter
#     def knowledge_transfer_folder(self, path):
#         assert os.path.isdir(path)
#         self._knowledge_transfer_folder = path
