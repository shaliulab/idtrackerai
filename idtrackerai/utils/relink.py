import os.path
import numpy as np


def relink(root_data_dir, experiment, session_number):
    print(session_number)
    session_number_formatted = str(session_number).zfill(6)
    
    video_file = os.path.join(
        root_data_dir,
        experiment,
        "idtrackerai",
        f"session_{session_number_formatted}",
        "video_object.npy"

    )
    video = np.load(video_file, allow_pickle=True)
