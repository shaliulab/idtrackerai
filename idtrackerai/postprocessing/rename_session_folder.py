from __future__ import absolute_import, print_function, division
import sys
import os
import numpy as np
from idtrackerai.blob import Blob
from idtrackerai.video import Video

if __name__ == "__main__":
    from idtrackerai.utils.GUI_utils import getInput, selectDir
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video_object = np.load(video_path).item(0)

    new_session_name = getInput("Rename sessions", "Select a new session name")
    video_object.rename_session_folder(new_session_name)
