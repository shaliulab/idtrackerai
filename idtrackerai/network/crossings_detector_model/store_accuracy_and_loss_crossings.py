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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logging.getLogger("__main__.store_accuracy_and_loss_crossings")

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
            ax2.set_xlabel('image type (individual or crossings)')
            ax2.set_ylabel('Per class \naccuracy')
            if index == 0:
                legend = ax1.legend()
                if legend_font_color is not None:
                    for text in legend.get_texts():
                        text.set_color(legend_font_color)
                ax1.set_ylabel('Accuracy')
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
            ax3.set_xlabel('image type (individual or crossings)')
            ax3.set_ylabel('Per class \naccuracy')
            if index == 0:
                legend = ax1.legend()
                if legend_font_color is not None:
                    for text in legend.get_texts():
                        text.set_color(legend_font_color)
                ax1.set_ylabel('Loss')
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Epochs')
            if plot_now:
                plt.draw()
                plt.pause(1e-8)

    def save(self):
        np.save(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), self.__dict__)

    def load(self):
        if os.path.isfile(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')):
            loss_accuracy_dictionary = np.load(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), allow_pickle=True).item()
            self.__dict__ = loss_accuracy_dictionary
