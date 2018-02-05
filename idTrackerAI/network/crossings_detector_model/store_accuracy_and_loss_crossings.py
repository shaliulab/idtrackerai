from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        # if network.is_restoring or network.is_knowledge_transfer:
        #     self.load()

    def append_data(self, loss_value, accuracy_value, individual_accuracy_value):
        self.loss.append(loss_value)
        self.accuracy.append(accuracy_value)
        self.individual_accuracy.append(individual_accuracy_value)

    def plot(self, axes_handles = None, index = 0, color = 'r', plot_now = True, legend_font_color = None, from_GUI = False):
        if from_GUI:
            ax1 = axes_handles[0]
            ax2 = axes_handles[1]
            ax1.plot(range(index,len(self.loss)),self.accuracy[index:], color,label=self.name)
            width = 0.35
            numAnimals = len(self.individual_accuracy[-1])
            if self.name == 'training':
                ind = np.arange(numAnimals) + 1 - width
            elif self.name == 'validation':
                ind = np.arange(numAnimals) + 1

            ax2.bar(ind, self.individual_accuracy[-1], width, color=color, alpha=0.4,label=self.name)
            ax2.set_xlabel('individual')
            ax2.set_ylabel('Individual accuracy')
            if index == 0:
                legend = ax1.legend()
                if legend_font_color is not None:
                    for text in legend.get_texts():
                        text.set_color(legend_font_color)
                ax1.set_ylabel('accuracy')
        else:
            ax1 = axes_handles[0]
            ax2 = axes_handles[1]
            ax3 = axes_handles[2]
            ax1.plot(range(index,len(self.loss)), self.loss[index:], color,label=self.name)
            ax2.plot(range(index,len(self.loss)),self.accuracy[index:], color,label=self.name)
            width = 0.35
            numAnimals = len(self.individual_accuracy[-1])
            if self.name == 'training':
                ind = np.arange(numAnimals) + 1 - width
            elif self.name == 'validation':
                ind = np.arange(numAnimals) + 1

            ax3.bar(ind, self.individual_accuracy[-1], width, color=color, alpha=0.4,label=self.name)
            ax3.set_xlabel('individual')
            ax3.set_ylabel('Individual accuracy')
            if index == 0:
                legend = ax1.legend()
                if legend_font_color is not None:
                    for text in legend.get_texts():
                        text.set_color(legend_font_color)
                ax1.set_ylabel('loss')
                ax2.set_ylabel('accuracy')
                ax2.set_xlabel('epochs')
            if plot_now:
                plt.draw()
                plt.pause(1e-8)

    def save(self):
        np.save(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), self.__dict__)

    def load(self):
        if os.path.isfile(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')):
            loss_accuracy_dictionary = np.load(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')).item()
            self.__dict__ = loss_accuracy_dictionary
