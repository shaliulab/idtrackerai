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

import logging
from confapp import conf

import torch
from torchvision import transforms

from idtrackerai.network.data_sets.identification_dataset import IdentificationDataset

logger = logging.getLogger("__main__.crossings_dataloader")


def get_training_data_loaders(video, train_data, val_data):


    logger.info("Creating training IdentificationDataset")
    training_set = IdentificationDataset(train_data,
                                         scope='training',
                                         transform=transforms.Compose(
                                           [transforms.ToTensor(),
                                            Normalize()])
                                         )
    train_loader = torch.utils.data.DataLoader(training_set,
                                               batch_size=conf.BATCH_SIZE_IDCNN,
                                               shuffle=False,
                                               num_workers=2)
    train_loader.num_classes = video.number_of_animals
    train_loader.image_shape = training_set[0][0].shape

    logger.info("Creating validation IdentificationDataset")
    validation_set = IdentificationDataset(val_data,
                                           scope='validation',
                                           transform=transforms.Compose(
                                             [transforms.ToTensor(),
                                              Normalize()])
                                        )
    val_loader = torch.utils.data.DataLoader(validation_set,
                                             batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
                                             shuffle=False,
                                             num_workers=2)
    val_loader.num_classes = video.number_of_animals
    val_loader.image_shape = validation_set[0][0].shape
    return train_loader, val_loader


def get_test_data_loader(test_data, number_of_classes):
    logger.info("Creating test IdentificationDataset")
    test_set = IdentificationDataset(test_data,
                                     scope='predict',
                                     transform=transforms.Compose(
                                             [transforms.ToTensor(),
                                              Normalize()]))
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=conf.BATCH_SIZE_PREDICTIONS_IDCNN,
                                              shuffle=False,
                                              num_workers=2)
    test_loader.num_classes = number_of_classes
    test_loader.image_shape = test_set[0][0].shape
    return test_loader


class Normalize(object):
    ### TODO: This is kind of a batch normalization but not trained. Explore using real BN in idCNN.
    def __init__(self, inplace=False):
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.tensor([tensor.mean()])
        std = torch.tensor([tensor.std()])
        return tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
