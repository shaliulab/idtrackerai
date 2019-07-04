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

import sys

import numpy as np
from tqdm import tqdm

from idtrackerai.list_of_fragments import ListOfFragments

def compute_model_velocity(fragments, number_of_animals, percentile = None):
    """computes the 2 * (percentile) of the distribution of velocities of identified fish.
    params
    -----
    blobs_in_video: list of blob objects
        collection of blobs detected in the video.
    number_of_animals int
    percentile int
    -----
    return
    -----
    float
    2 * np.max(distance_travelled_in_individual_fragments) if percentile is None
    2 * percentile(velocity distribution of identified animals) otherwise
    """
    distance_travelled_in_individual_fragments = []

    for fragment in tqdm(fragments, desc = "computing velocity model"):
        if fragment.is_an_individual:
            distance_travelled_in_individual_fragments.extend(fragment.frame_by_frame_velocity())

    return 2 * np.max(distance_travelled_in_individual_fragments) if percentile is None else 2 * np.percentile(distance_travelled_in_individual_fragments, percentile)
