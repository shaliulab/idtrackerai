import os
import subprocess
from glob import glob
from pprint import pprint
from idtrackerai.constants import COMPRESSED_VIDEO_PATH

dir_name = os.path.dirname(os.path.realpath(__file__))

test_folders = glob(os.path.join(dir_name, "test_*"))

outputs = {}
for test_folder in test_folders:
    print(test_folder)
    json_file = os.path.join(test_folder, "test.json")
    command = (
        f"idtrackerai terminal_mode "
        f"--load {json_file} "
        f"--exec track_video "
        f"--_video_path {COMPRESSED_VIDEO_PATH}"
    )
    output = subprocess.run(
        command.split(" "),
        check=True,
        capture_output=True,
        text=True,
    )
    if "Success" in output.stderr.splitlines()[-1]:
        outputs[test_folder] = "worked"
    else:
        outputs[test_folder] = "failed"

    with open(os.path.join(test_folder, "output.txt"), "w") as file:
        file.write(output.stderr)

pprint(outputs)
