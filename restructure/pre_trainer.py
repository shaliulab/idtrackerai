import os
import sys
sys.path.append('./network')

from id_CNN import ConvNetwork
from video import Video

class Network_Params(object):
    def __init__(video,
                learning_rate, keep_prob,
                use_adam_optimiser, scopes_layers_to_optimize,
                restore_folder = None, save_folder = None, knowledge_transfer_folder = None):

        self.number_of_animals = video.number_of_animals
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self._restore_folder = restore_folder
        self._save_folder = save_folder
        self._knowledge_transfer_folder = knowledge_transfer_folder
        self.use_adam_optimiser = use_adam_optimiser
        self.scopes_layers_to_optimize = scopes_layers_to_optimize

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

class Get_Data(object):
    def __init__(self):
        pass


def pre_train(global_fragments, network_params):
    #a high distance travelled corresponds to higher variability in the images
    #and longer global fragments, so:
    order_global_fragments_by_distance_travelled(global_fragments)

    net = ConvNetwork(network_params)
