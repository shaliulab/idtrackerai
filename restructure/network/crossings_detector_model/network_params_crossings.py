from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import sys
sys.path.append('./utils')
# from cnn_utils import *

import numpy as np

class NetworkParams_crossings(object):
    def __init__(self,number_of_classes,
                architecture = None,
                learning_rate = None, keep_prob = None,
                use_adam_optimiser = False, scopes_layers_to_optimize = None,
                restore_folder = None, save_folder = None, knowledge_transfer_folder = None,
                image_size = None):

        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.keep_prob = keep_prob
        self._restore_folder = restore_folder
        self._save_folder = save_folder
        print("(in network params) self.params._save_folder", self._save_folder)
        self._knowledge_transfer_folder = knowledge_transfer_folder
        self.use_adam_optimiser = use_adam_optimiser
        self.image_size = image_size

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
