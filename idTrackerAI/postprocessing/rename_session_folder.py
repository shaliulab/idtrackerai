from __future__ import absolute_import, print_function, division
import sys
sys.path.append('../utils')
sys.path.append('../')
import os
import numpy as np
from GUI_utils import rename_session_folder, getInput, selectDir
from blob import Blob

session_path = selectDir('./') #select path to video
video_path = os.path.join(session_path,'video_object.npy')
print("loading video object...")
video_object = np.load(video_path).item(0)

new_session_name = getInput("Rename sessions", "Select a new session name")
rename_session_folder(video_object, new_session_name)
