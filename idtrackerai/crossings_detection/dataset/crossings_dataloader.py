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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import logging

import torch
from confapp import conf
from torchvision import transforms

from idtrackerai.crossings_detection.dataset.crossings_dataset import (
    CrossingDataset,
)

logger = logging.getLogger("__main__.crossings_dataloader")

if os.name == "nt":  # windows
    # Using multipricessing in Windows causes a
    # recursion limit error difficut to debug
    num_workers_train = 0
    num_workers_val = 0
else:
    num_workers_train = 4
    num_workers_val = 4


def get_training_data_loaders(video, train_blobs, val_blobs):
    logger.info("Creating training and validation data loaders")
    training_set = CrossingDataset(
        train_blobs,
        video,
        scope="training",
        transform=transforms.Compose([transforms.ToTensor(), Normalize()]),
    )
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=conf.BATCH_SIZE_DCD,
        shuffle=False,
        num_workers=num_workers_train,
    )
    train_loader.num_classes = 2
    train_loader.image_shape = training_set[0][0].shape

    logger.info("Creating validation CrossingDataset")
    validation_set = CrossingDataset(
        val_blobs,
        video,
        scope="validation",
        transform=transforms.Compose([transforms.ToTensor(), Normalize()]),
    )
    val_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_DCD,
        shuffle=False,
        num_workers=num_workers_val,
    )
    val_loader.num_classes = 2
    val_loader.image_shape = validation_set[0][0].shape
    return train_loader, val_loader


def get_test_data_loader(video, test_blobs):
    logger.info("Creating test CrossingDataset")
    test_set = CrossingDataset(
        test_blobs,
        video,
        scope="test",
        transform=transforms.Compose([transforms.ToTensor(), Normalize()]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=conf.BATCH_SIZE_PREDICTIONS_DCD,
        shuffle=False,
        num_workers=num_workers_val,
    )
    test_loader.num_classes = 2
    test_loader.image_shape = test_set[0][0].shape
    return test_loader


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

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
        # return F.normalize(tensor, tensor.mean(), tensor.std(), self.inplace)
