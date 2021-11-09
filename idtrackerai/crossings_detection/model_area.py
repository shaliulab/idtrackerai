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

from confapp import conf
import numpy as np
import logging

from idtrackerai.list_of_blobs import ListOfBlobs

logger = logging.getLogger("__main__.model_area")


class ModelArea(object):
    """Model of the area used to perform a first discrimination between blobs
    representing single individual and multiple touching animals (crossings)

    Attributes
    ----------

    median : float
        median of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    mean : float
        mean of the area of the blobs segmented from portions of the video in
        which all the animals are visible (not touching)
    std : float
        standard deviation of the area of the blobs segmented from portions of
        the video in which all the animals are visible (not touching)
    std_tolerance : int
        tolerance factor

    Methods
    -------
    __call__:
      some description
    """

    def __init__(self, mean, median, std):
        self.median = median
        self.mean = mean
        self.std = std
        self.std_tolerance = conf.MODEL_AREA_SD_TOLERANCE

    def __call__(self, area, std_tolerance=None):
        if std_tolerance is not None:
            self.std_tolerance = std_tolerance
        return bool((area - self.median) < self.std_tolerance * self.std)


def compute_model_area_and_body_length(
    list_of_blobs: ListOfBlobs, number_of_animals: int
):
    """computes the median and standard deviation of the area of all the blobs
    in the the video and the median of the the diagonal of the bounding box.
    """
    # areas are collected throughout the entire video in the cores of the global fragments
    areas_and_body_length = np.asarray(
        [
            (blob.area, blob.estimated_body_length)
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if len(blobs_in_frame) == number_of_animals
        ]
    )
    logger.info(f"Model area computed with {len(areas_and_body_length)}")
    if areas_and_body_length.shape[0] == 0:
        raise ValueError(
            "There is not part in the video where the {} "
            "animals are visible. "
            "Try a different segmentation or check the "
            "number of animals in the video.".format(number_of_animals)
        )
    median_area = np.median(areas_and_body_length[:, 0])
    mean_area = np.mean(areas_and_body_length[:, 0])
    std_area = np.std(areas_and_body_length[:, 0])
    median_body_length = np.median(areas_and_body_length[:, 1])
    return ModelArea(mean_area, median_area, std_area), median_body_length
