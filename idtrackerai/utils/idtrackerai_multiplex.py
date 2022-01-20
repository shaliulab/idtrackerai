"""
Functionality to quickly check the accuracy of many sessions
"""
import argparse
import logging
import re
import os.path

from imgstore.multistores import MultiStore
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
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
    incomplete = list_uncomplete_frames_and_blobs(session_folder)
    check = {
        accuracy[0]: accuracy[1]
    }
    # incomplete_frames = incomplete.pop("incomplete_frames")
    # incomplete["incomplete_frames"] = len(incomplete_frames)

    for name, val in incomplete:
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
    """
    
    """

    incomplete_frames_list = []
    blobs_file = pick_blob_collection(session_folder)
    video_object_path = os.path.join(session_folder, "video_object.npy")
    video = np.load(video_object_path, allow_pickle=True).item()
    number_of_animals = video.user_defined_parameters["number_of_animals"]

    incomplete_blobs = 0
    finally_incomplete_blobs = 0

    list_of_blobs = ListOfBlobs.load(blobs_file)
    for frame_in_chunk, blobs_in_frame in enumerate(list_of_blobs.blobs_in_video):
        if len(blobs_in_frame) != number_of_animals:
            assert frame_in_chunk == blobs_in_frame[0].frame_number
            incomplete_frames_list.append(frame_in_chunk)

        for blob in blobs_in_frame:
            if blob.identity is None:
                incomplete_blobs += 1

            if blob.final_identities is None:
                finally_incomplete_blobs += 1


    return (
        ("incomplete_frames", incomplete_frames_list),
        ("incomplete_blob", incomplete_blobs),
        ("finally_incomplete_blobs", finally_incomplete_blobs)
    )
            


def validate_sessions(interval, **kwargs):

    ok_folders = []
    for i in range(*interval):
        session_folder = f"session_{str(i).zfill(6)}"
        if is_idtrackerai_folder(session_folder, **kwargs):
            ok_folders.append(session_folder)
        else:
            logger.warning(f"{session_folder} is corrupted")

    return ok_folders


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("interval", nargs="+", type=int)
    ap.add_argument("--n-jobs", dest="n_jobs", default=1, type=int)
    ap.add_argument("--trajectories", action="store_true", dest="trajectories", default=True)
    ap.add_argument("--no-trajectories", action="store_false", dest="trajectories", default=True)
    return ap


def anti_merge(df1, df2):
    data_all = df1.reset_index().merge(df2.reset_index()["index"], how="left", on="index", indicator=True)
    data_all = data_all.loc[data_all["_merge"] == "left_only"]
    pd.concat()


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        ap.parse_args()

    args = ap.parse_args()

    assert os.path.exists("cross_index.csv"), "Please generate the crossindex.csv file with multistore-index --input metadata.yaml"
    crossindex = pd.read_csv("cross_index.csv", index_col=0)

    folders = validate_sessions(args.interval, trajectories=args.trajectories)
    # accuracies = list_accuracy_all(folders)
    validation = check_idtrackerai_results(folders, n_jobs=args.n_jobs)
    print(f"session_folder{SEP}accuracy{SEP}incomplete_frames{SEP}incomplete_blobs{SEP}finally_incomplete_blobs")

    incomplete_frames = {"main": {}, "delta": {}}

    logger.info("Opening store")
    store = MultiStore.new_for_filename(
        "metadata.yaml",
        ref_chunk=1,
        chunk_numbers = range(*args.interval)
    )


    for session_folder, data in validation.items():
        print(session_folder, end="")
        for stat in data.items():
            name, value = stat
            if name == "incomplete_frames":
                incomplete_frames["main"][session_folder] = value
                value = len(value)

            print(SEP, end="")
            print(f"{value}", end = "")
        
        print()


    for session, frames in incomplete_frames["main"].items():
        chunk = int(re.match("session_(.*)", session).group(1))
        frame_number_in_delta_time = [
            crossindex.loc[
                np.bitwise_and(
                    crossindex["main_chunk"] == chunk,
                    crossindex["main_frame_idx"] == frame_idx,
                ),
                "delta_number"
            ].values.tolist()[0]
            for frame_idx in frames
        ]

        incomplete_frames["delta"][session] = frame_number_in_delta_time

    
    for session in incomplete_frames["main"]:
        to_export = pd.merge(
            pd.DataFrame(incomplete_frames["main"][session], columns=["main"]),
            pd.DataFrame(incomplete_frames["delta"][session], columns=["delta"]),
            left_index=True, right_index=True
        )

        to_export.to_csv(f"{session}_missing-frames.csv")









 


if __name__ == "__main__":
    main()
