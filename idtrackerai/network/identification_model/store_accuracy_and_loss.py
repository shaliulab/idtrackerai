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

import numpy as np
import matplotlib.pyplot as plt

from idtrackerai.utils.py_utils import  get_spaced_colors_util

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.store_accuracy_and_loss")

class Store_Accuracy_and_Loss(object):
    """Store the loss, accuracy and individual accuracy values computed during
    training and validation

    Parameters
    ----------
    _path_to_accuracy_error_data :
        Path to save the lists :attr:`loss`, :attr:`accuracy`, :attr:`individual_accuracy`
    name : string
        'Training' or 'Validation'
    loss : list
        List with the values of the loss
    accuracy : list
        List with the values of the accuracy
    individual_accuracy : list
        List with the values of the individual accuracies
    number_of_epochs_completed :
        Number of epochs completed
    scope : string
        'Training' if the class is instantiated during the training of the accumulation.
        'Pretraining' if the class is instantiated during the training
    """
    def __init__(self, network, name, scope = None):
        self._path_to_accuracy_error_data = network.params.save_folder
        self.name = name
        self.loss = []
        self.accuracy = []
        self.individual_accuracy = []
        self.number_of_epochs_completed = []
        self.scope = scope
        self.load()

    def append_data(self, loss_value, accuracy_value, individual_accuracy_value):
        """Appends the `loss_value`, `accuracy_value` and `individual_accuracy_value`
        to their correspoding lists
        """
        self.loss.append(loss_value)
        self.accuracy.append(accuracy_value)
        self.individual_accuracy.append(individual_accuracy_value)

    def plot(self, axes_handles = None, index = 0, color = 'r', canvas_from_GUI = None, legend_font_color = None):
        """Plots the accuracy and the individual accuracies for every epoch
        """
        if canvas_from_GUI is not None:
            ax1 = axes_handles[0]
            ax2 = axes_handles[1]
            ax1.plot(range(len(self.loss)),self.accuracy, color,label=self.name)
            width = 0.35
            numAnimals = len(self.individual_accuracy[-1])
            if self.name == 'training':
                ind = np.arange(numAnimals) + 1 - width
            elif self.name == 'validation':
                ind = np.arange(numAnimals) + 1
            ax1.set_xlabel('Epochs')
            ax2.bar(ind, self.individual_accuracy[-1], width, color=color, alpha=0.4,label=self.name)
            ax2.set_xlabel('Individual')
            ax2.set_ylabel('Individual \naccuracy')
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
            ax2.set_xlabel('Epochs')
            width = 0.35
            numAnimals = len(self.individual_accuracy[-1])
            if self.name == 'training':
                ind = np.arange(numAnimals)+1-width
            elif self.name == 'validation':
                ind = np.arange(numAnimals)+1
            ax3.bar(ind, self.individual_accuracy[-1], width, color=color, alpha=0.4,label=self.name)
            ax3.set_xlabel('Individual')
            ax3.set_ylabel('Individual \naccuracy')
            if index == 0:
                legend = ax1.legend()
                if legend_font_color is not None:
                    for text in legend.get_texts():
                        text.set_color(legend_font_color)
                ax1.set_ylabel('Loss')
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('Epochs')
        if canvas_from_GUI is None:
            plt.draw()
            plt.pause(1e-8)
        else:
            canvas_from_GUI.draw()

    def plot_global_fragments(self, ax_handles, video, fragments, black = False, canvas_from_GUI = None):
        """Plots the global fragments used for training until the current epoch
        """
        import matplotlib.patches as patches
        if canvas_from_GUI is not None:
            ax4 = ax_handles[2]
            ax4.cla()
        else:
            ax4 = ax_handles[3]
            ax4.cla()
        colors = get_spaced_colors_util(video.number_of_animals, norm=True, black=black)
        attribute_to_check = 'used_for_training' if self.scope == 'training' else 'used_for_pretraining'

        for fragment in fragments:
            if getattr(fragment, attribute_to_check) and fragment.is_an_individual:
                if fragment.final_identities is not None and len(fragment.final_identities)==1:
                    blob_index = fragment.final_identities[0] - 1
                elif self.scope == 'pretraining':
                    blob_index = fragment.blob_hierarchy_in_starting_frame
                else:
                    blob_index = fragment.temporary_id
                (start, end) = fragment.start_end
                ax4.add_patch(
                    patches.Rectangle(
                        (start, blob_index - 0.5),   # (x,y)
                        end - start,  # width
                        1.,          # height
                        fill=True,
                        edgecolor=None,
                        facecolor=colors[fragment.temporary_id if attribute_to_check == 'used_for_training' else fragment._temporary_id_for_pretraining],
                        alpha = 1.
                    )
                )
        ax4.axis('tight')
        ax4.set_xlabel('Frame number')
        ax4.set_ylabel('Blob index')
        if not hasattr(video, 'maximum_number_of_blobs') or video.maximum_number_of_blobs == 0:
            max_y = video.number_of_animals
        else:
            max_y = video.maximum_number_of_blobs
        ax4.set_yticks(range(0, max_y, 3))
        ax4.set_yticklabels(range(1, max_y + 1, 3))
        ax4.set_xlim([0., video.number_of_frames])
        ax4.set_ylim([-.5, max_y + .5 - 1])

    def save(self, number_of_epochs_completed):
        """Saves the values stored
        """
        self.number_of_epochs_completed.append(number_of_epochs_completed)
        np.save(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), self.__dict__)

    def load(self):
        """Load the values stored in case there are any saved
        """
        if os.path.isfile(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')):
            loss_accuracy_dictionary = np.load(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), allow_pickle=True).item()
            self.__dict__ = loss_accuracy_dictionary
            if not hasattr(self, 'number_of_epochs_completed'):
                self.number_of_epochs_completed = []
