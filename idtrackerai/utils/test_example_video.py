import os
import sys
import argparse
import json

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

    output_file_avi = os.path.join(video_example_folder, 'example_video_idtrackerai.avi')

    gdown.download(TEST_VIDEO_URL, output_file_avi, quiet=False)

    return output_file_avi


def update_json(video_path, args):
    json_file_path = os.path.join(IDTRACKERAI_FOLDER, 'utils', 'test.json')

    with open(json_file_path) as json_file:
        json_content = json.load(json_file)

    json_content['_video']['value'] = video_path
    if args.no_identities:
        json_content['_no_ids']['value'] = "True"

    with open(json_file_path, 'w') as json_file:
        json.dump(json_content, json_file)

    return json_file_path


def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder",
                        type=str,
                        help="Path to the folder where the video will be stored")
    parser.add_argument("-n", "--no_identities",
                        action='store_true',
                        help="Flag to track without identities")
    args = parser.parse_args()
    video_path = download_example_video(args.output_folder)
    json_file_path = update_json(video_path, args)
    os.system('idtrackerai terminal_mode --load {} --exec track_video'.format(json_file_path))


if __name__ == "__main__":
    test()
