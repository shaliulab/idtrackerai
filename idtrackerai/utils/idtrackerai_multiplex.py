"""
Functionality to quickly check the accuracy of many sessions
"""
import argparse
import logging
import re
import os.path

import joblib
import numpy as np
from tqdm import tqdm

from idtrackerai.utils.py_utils import is_idtrackerai_folder, pick_blob_collection
from idtrackerai.list_of_blobs import ListOfBlobs

logger = logging.getLogger(__name__)

SEP = "\t"

def list_accuracy(session_folder):
    accuracy = None
    logfile = session_folder + "_error.txt"
    with open(logfile, "r") as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            match = re.match(
                r".*Estimated accuracy: ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?).*",
                line,
            )
            if match:
                accuracy = round(float(match.group(1)), 3)
    return ("accuracy", accuracy)


def check_session_folder(session_folder):
    check = {}
    accuracy = list_accuracy(session_folder)
    uncomplete = list_uncomplete_frames_and_blobs(session_folder)
    check = {
        accuracy[0]: accuracy[1]
    }
    for name, val in uncomplete:
        check[name] = val
    
    return (session_folder, check)

        
def check_idtrackerai_results(folders, n_jobs=1):

    check = {}
    if n_jobs == 1:

        for session_folder in tqdm(folders):
            _, check_sf = check_session_folder(session_folder)
            check[session_folder] = check_sf
    
    else:
        output = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(check_session_folder)(
                session_folder
            ) for session_folder in folders
        )
        for folder, check_sf in output:
            check[folder] = check_sf
        
    return check

def list_uncomplete_frames_and_blobs(session_folder):

    uncomplete_frames = []
    blobs_file = pick_blob_collection(session_folder)
    video_object_path = os.path.join(session_folder, "video_object.npy")
    video = np.load(video_object_path, allow_pickle=True).item()
    number_of_animals = video.user_defined_parameters["number_of_animals"]

    incomplete_frames = 0
    incomplete_blobs = 0
    finally_incomplete_blobs = 0

    list_of_blobs = ListOfBlobs.load(blobs_file)
    for frame_in_chunk, blobs_in_frame in enumerate(list_of_blobs.blobs_in_video):
        if len(blobs_in_frame) != number_of_animals:
            incomplete_frames += 1

        for blob in blobs_in_frame:
            if blob.identity is None:
                incomplete_blobs += 1

            if blob.final_identities is None:
                finally_incomplete_blobs += 1


    return (
        ("incomplete_frames", incomplete_frames),
        ("incomplete_blob", incomplete_blobs),
        ("finally_incomplete_blobs", finally_incomplete_blobs)
        )
            


def validate_sessions(interval):

    ok_folders = []
    for i in range(*interval):
        session_folder = f"session_{str(i).zfill(6)}"
        if is_idtrackerai_folder(session_folder):
            ok_folders.append(session_folder)
        else:
            logger.warning(f"{session_folder} is corrupted")

    return ok_folders


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("interval", nargs="+", type=int)
    ap.add_argument("--n-jobs", dest="n_jobs", default=1, type=int)

    return ap


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        ap.parse_args()

    args = ap.parse_args()

    folders = validate_sessions(args.interval)
    # accuracies = list_accuracy_all(folders)
    validation = check_idtrackerai_results(folders, n_jobs=args.n_jobs)
    print(f"session_folder{SEP}accuracy{SEP}incomplete_frames{SEP}incomplete_blobs{SEP}finally_incomplete_blobs")
    for session_folder, data in validation.items():
        print(session_folder, end="")
        for stat in data.items():
            name, value = stat
            print(SEP, end="")
            print(f"{value}", end = "")
        
        print()

if __name__ == "__main__":
    main()
