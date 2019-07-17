import os
import sys
import argparse

import gdown

from idtrackerai.constants import IDTRACKERAI_FOLDER, TEST_VIDEO_URL

def create_data_folder():
    idtrackerai_data_folder = os.path.join(os.path.dirname(IDTRACKERAI_FOLDER), 'data')
    if not os.path.isdir(idtrackerai_data_folder):
        os.mkdir(idtrackerai_data_folder)

    return idtrackerai_data_folder


def download_example_video(output_folder=''):
    if not output_folder:
        output_folder = create_data_folder()

    video_example_folder = os.path.join(output_folder, 'idtrackerai_example_video')
    if not os.path.isdir(video_example_folder):
        os.mkdir(video_example_folder)

    output_file = os.path.join(video_example_folder, 'example_video_idtrackerai.avi')

    gdown.download(TEST_VIDEO_URL, output_file, quiet=False)


def download():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder",
                        type=str,
                        help="Path to the folder where the video will be stored")
    args = parser.parse_args()
    download_example_video(args.output_folder)


if __name__ == "__main__":
    download()
