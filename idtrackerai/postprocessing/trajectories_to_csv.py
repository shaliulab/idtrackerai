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

import json
import os
import sys

import numpy as np


def save_array_to_csv(path, array, key=""):
    array = np.squeeze(array)
    if array.ndim == 3:
        array_reshaped = np.reshape(
            array, (-1, array.shape[1] * array.shape[2])
        )
        array_header = ",".join(
            [
                coord + str(i)
                for i in range(1, array.shape[1] + 1)
                for coord in ["x", "y"]
            ]
        )
        np.savetxt(path, array_reshaped, delimiter=",", header=array_header)

    elif array.ndim == 2:
        array_header = ",".join(
            [key + str(i) for i in range(1, array.shape[1] + 1)]
        )
        np.savetxt(path, array, delimiter=",", header=array_header)


def convert_trajectories_file_to_csv_and_json(trajectories_file):
    trajectories_dict = np.load(trajectories_file, allow_pickle=True).item()

    file_name = os.path.splitext(trajectories_file)[0]

    if trajectories_dict is None:
        trajectories_dict = np.load(
            trajectories_file, allow_pickle=True
        ).item()

    attributes_dict = {}
    for key in trajectories_dict:
        if isinstance(trajectories_dict[key], np.ndarray):
            key_csv_file = file_name + "." + key + ".csv"
            save_array_to_csv(key_csv_file, trajectories_dict[key], key=key)

        else:
            attributes_dict[key] = trajectories_dict[key]

    attributes_file_name = file_name + ".attributes.json"
    with open(attributes_file_name, "w") as fp:
        json.dump(attributes_dict, fp)


if __name__ == "__main__":
    path = sys.argv[1]

    for root, folders, files in os.walk(path):
        for file in files:
            if "trajectories" in file and ".npy" in file:
                trajectories_file = os.path.join(root, file)
                print(
                    "Converting {} to .csv and .json".format(trajectories_file)
                )
                convert_trajectories_file_to_csv_and_json(trajectories_file)
