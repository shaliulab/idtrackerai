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

import argparse
import json
import os
import shutil

import gdown

from idtrackerai.constants import (
    IDTRACKERAI_FOLDER,
    TEST_VIDEO_URL,
    COMPRESSED_VIDEO_PATH,
)


def create_data_folder():
    idtrackerai_data_folder = os.path.join(
        os.path.dirname(IDTRACKERAI_FOLDER), "data"
    )
    if not os.path.isdir(idtrackerai_data_folder):
        os.mkdir(idtrackerai_data_folder)

    return idtrackerai_data_folder


def download_example_video(output_folder=""):
    if not output_folder:
        output_folder = create_data_folder()

    video_example_folder = os.path.join(
        output_folder, "idtrackerai_example_video"
    )
    if not os.path.isdir(video_example_folder):
        os.mkdir(video_example_folder)

    output_file_avi = os.path.join(
        video_example_folder, "example_video_idtrackerai.avi"
    )
    if not os.path.isfile(output_file_avi):
        gdown.download(TEST_VIDEO_URL, output_file_avi, quiet=False)

    return output_file_avi


def generate_json_file(video_path, args):
    default_json_file_path = os.path.join(
        IDTRACKERAI_FOLDER, "utils", "test.json"
    )
    updated_json_file = os.path.join(IDTRACKERAI_FOLDER, "utils", "test_.json")

    with open(default_json_file_path) as json_file:
        json_content = json.load(json_file)

    json_content["_video"]["value"] = video_path
    if args.no_identities:
        json_content["_no_ids"]["value"] = "True"
    else:
        json_content["_no_ids"]["value"] = "False"

    with open(updated_json_file, "w") as json_file:
        json.dump(json_content, json_file)

    return updated_json_file


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="Path to the folder where the video will be stored",
    )
    parser.add_argument(
        "-n",
        "--no_identities",
        action="store_true",
        help="Flag to track without identities",
    )
    args = parser.parse_args()

    if args.output_folder:
        print("Copying test video file to: {args.output_folder}")
        _, video_name = os.path.split(COMPRESSED_VIDEO_PATH)
        video_path = os.path.join(args.output_folder, video_name)
        shutil.copyfile(COMPRESSED_VIDEO_PATH, video_path)
    else:
        video_path = COMPRESSED_VIDEO_PATH

    json_file_path = generate_json_file(video_path, args)
    command = (
        f"idtrackerai terminal_mode --load {json_file_path} --exec track_video"
    )
    print("Executting command: \n")
    print(command)
    os.system(command)


if __name__ == "__main__":
    test()
