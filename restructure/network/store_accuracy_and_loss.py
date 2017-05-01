from __future__ import absolute_import, division, print_function
import os
import numpy as np

class Store_Accuracy_and_Loss(object):
    """Store the loss, accuracy and individual accuracy values computed during
    training and validation
    """
    def __init__(self, network, name):
        self._path_to_accuracy_error_data = network.params.save_folder
        self.name = name
        self.loss = []
        self.accuracy = []
        self.individual_accuracy = []
        if network.is_restoring or network.is_knowledge_transfer:
            self.load()

    def append_data(self, loss_value, accuracy_value, individual_accuracy_value):
        self.loss.append(loss_value)
        self.accuracy.append(accuracy_value)
        self.individual_accuracy.append(individual_accuracy_value)

    def save(self):
        np.save(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), self.__dict__)

    def load(self):
        if os.path.isfile(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')):
            loss_accuracy_dictionary = np.load(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')).item()
            self.__dict__ = loss_accuracy_dictionary
