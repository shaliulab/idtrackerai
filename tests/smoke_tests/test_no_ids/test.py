import os

from idtrackerai.constants import COMPRESSED_VIDEO_PATH

dir_name = os.path.dirname(os.path.realpath(__file__))
json_file = os.path.join(dir_name, "test.json")
command = (
    f"idtrackerai terminal_mode "
    f"--load {json_file} "
    f"--exec track_video "
    f"--_video_path {COMPRESSED_VIDEO_PATH} "
)
print("Executting command: \n")
print(command)
os.system(command)
