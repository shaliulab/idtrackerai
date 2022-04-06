import argparse
import glob
import os.path

import numpy as np
from idtrackerai.postprocessing.individual_videos import (
    generate_individual_videos,
)

def get_video_object_and_trajectories(video_path, session_name):
    video_folder = os.path.dirname(video_path)
    session_folder = "session_{0}".format(session_name)
    session_path = os.path.join(video_folder, session_folder)
    trajs_wo_path = os.path.join(session_path, "trajectories_wo_gaps")
    trajs_path = os.path.join(session_path, "trajectories")
    traj_wo_ids_path = os.path.join(
        session_path, "trajectories_wo_identification"
    )

    video_object = np.load(
        os.path.join(session_path, "video_object.npy"), allow_pickle=True
    ).item()

    if os.path.exists(trajs_wo_path):
        trajectories_file = glob.glob(os.path.join(trajs_wo_path, "*.npy"))[-1]
    elif os.path.exists(trajs_path):
        trajectories_file = glob.glob(os.path.join(trajs_path, "*.npy"))[-1]
    elif os.path.exists(traj_wo_ids_path):
        trajectories_file = glob.glob(os.path.join(traj_wo_ids_path, "*.npy"))[
            -1
        ]

    trajectories = np.load(trajectories_file, allow_pickle=True).item()[
        "trajectories"
    ]

    return video_object, trajectories

def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-path", dest="video_path", required=True, type=str)
    ap.add_argument("--session", dest="session", required=True, type=int)
    return ap

def main(args=None):
    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    session = str(args.session).zfill(6)
    video_object, trajectories = get_video_object_and_trajectories(
        args.video_path, session
    )
    generate_individual_videos(video_object, trajectories)


if __name__ == "__main__":
    main()
