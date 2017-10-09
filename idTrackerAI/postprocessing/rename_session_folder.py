from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../utils')
sys.path.append('../')
import os
import numpy as np
from GUI_utils import getInput, selectDir
from blob import Blob
from video import Video

session_path = selectDir('./') #select path to video
video_path = os.path.join(session_path,'video_object.npy')
print("loading video object...")
video_object = np.load(video_path).item(0)

new_session_name = getInput("Rename sessions", "Select a new session name")
video_object.rename_session_folder(new_session_name)
