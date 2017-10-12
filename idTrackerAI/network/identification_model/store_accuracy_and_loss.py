from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./utils')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from py_utils import get_spaced_colors_util

class Store_Accuracy_and_Loss(object):
    """Store the loss, accuracy and individual accuracy values computed during
    training and validation
    """
    def __init__(self, network, name, scope = None):
        self._path_to_accuracy_error_data = network.params.save_folder
        self.name = name
        self.loss = []
        self.accuracy = []
        self.individual_accuracy = []
        self.scope = scope
        if network.is_restoring or network.is_knowledge_transfer:
            self.load()

    def append_data(self, loss_value, accuracy_value, individual_accuracy_value):
        self.loss.append(loss_value)
        self.accuracy.append(accuracy_value)
        self.individual_accuracy.append(individual_accuracy_value)

    def plot(self, axes_handles = None, index = 0, color = 'r'):
        ax1 = axes_handles[0]
        ax2 = axes_handles[1]
        ax3 = axes_handles[2]
        ax1.plot(range(index,len(self.loss)), self.loss[index:], color,label=self.name)
        ax2.plot(range(index,len(self.loss)),self.accuracy[index:], color,label=self.name)
        width = 0.35
        numAnimals = len(self.individual_accuracy[-1])
        if self.name == 'training':
            ind = np.arange(numAnimals)+1-width
        elif self.name == 'validation':
            ind = np.arange(numAnimals)+1
        ax3.bar(ind, self.individual_accuracy[-1], width, color=color, alpha=0.4,label=self.name)
        ax3.set_xlabel('individual')
        ax3.set_ylabel('Individual accuracy')
        if index == 0:
            ax1.legend()
            ax1.set_ylabel('loss')
            ax2.set_ylabel('accuracy')
            ax2.set_xlabel('epochs')
        plt.draw()
        plt.pause(1e-8)

    def plot_global_fragments(self, ax_handles, video, fragments, black = False):
        import matplotlib.patches as patches
        ax4 = ax_handles[3]
        ax4.cla()
        colors = get_spaced_colors_util(video._maximum_number_of_blobs, norm=True, black=black)
        attribute_to_check = 'used_for_training' if self.scope == 'training' else 'used_for_pretraining'
        for fragment in fragments:
            if getattr(fragment, attribute_to_check):
                blob_index = fragment.blob_hierarchy_in_starting_frame
                (start, end) = fragment.start_end
                ax4.add_patch(
                    patches.Rectangle(
                        (start, blob_index - 0.5),   # (x,y)
                        end - start,  # width
                        1.,          # height
                        fill=True,
                        edgecolor=None,
                        facecolor=colors[fragment.temporary_id if attribute_to_check == 'used_for_training' else int(blob_index)],
                        alpha = 1.
                    )
                )

        ax4.axis('tight')
        ax4.set_xlabel('Frame number')
        ax4.set_ylabel('Blob index')
        ax4.set_yticks(range(0,video.number_of_animals,4))
        ax4.set_yticklabels(range(1,video.number_of_animals+1,4))
        ax4.set_xlim([0., video.number_of_frames])
        ax4.set_ylim([-.5, .5 + video.number_of_animals])

    def save(self):
        np.save(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy'), self.__dict__)

    def load(self):
        if os.path.isfile(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')):
            loss_accuracy_dictionary = np.load(os.path.join(self._path_to_accuracy_error_data, self.name + '_loss_acc_dict.npy')).item()
            self.__dict__ = loss_accuracy_dictionary
