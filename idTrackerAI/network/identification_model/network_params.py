from __future__ import absolute_import, division, print_function
import os
import logging

logger = logging.getLogger("__main__.network_params")

class NetworkParams(object):
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

    def check_identity_transfer_consistency(self, knowledge_transfer_info_dict):
        print("***************************************************************")
        if knowledge_transfer_info_dict['number_of_animals'] != self.number_of_animals:
            logger.info('It is not yet possible to transfer the identity because the number of ' + \
                        'animals in the video is different from the number of ouput units in the last layer of the model selected. It will be implemented in the future.')
            logger.info('Only the knowledge from the convolutional filters will be transferred')
            self.knowledge_transfer_folder = self.restore_folder
            self.restore_folder = None
        elif knowledge_transfer_info_dict['input_image_size'][2] != self.image_size[2]:
            logger.info('It is not yet possible to transfer the identity because the number of ' + \
                        'channels in the video is different from the one declared in the model selected.')
            raise ValueError('The algorithm cannot proceed.')
        elif self.image_size[0] != self.image_size[1] or knowledge_transfer_info_dict['input_image_size'][0] != knowledge_transfer_info_dict['input_image_size'][1]:
            raise ValueError('The algorithm works with square images. Either the input image or the input of the selected model are not square')
        elif knowledge_transfer_info_dict['input_image_size'] != self.image_size:
            self.target_image_size = knowledge_transfer_info_dict['input_image_size']
            ratio_image_size = self.target_image_size[0]/self.image_size[0]
            if ratio_image_size >= .9 and ratio_image_size <= 1.1:
                # we pad or crop
                logger.info('The ratio target_image_size/input_image is %.2f, we pad with zeros or crop centrally the input image to match the target size' %ratio_image_size)
                self.action_on_image = 'pad_or_crop'

            elif ratio_image_size > 1.1 or ratio_image < .9:
                # we resize and (pad or crop)
                if ratio_image < 1:
                    self.pre_target_image_size = (int(self.target_image_size * .9),) * 2
                elif ratio_image > 1:
                    self.pre_target_image_size = (int(self.target_image_size * 1.1),) * 2
                    self.action_on_image = 'resize_and_pad_or_crop'
                logger.info('The ratio target_image_size/input_image is %.2f, we resize and pad with zeros (or crop centrally) the input image to match the size' %ratio_image_size)
        print("image_size ", self.image_size)
        print("target_image_size, ", self.target_image_size)
        print("pre_target_image_size ", self.pre_target_image_size)
        print("***************************************************************")
