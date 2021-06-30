import os
import argparse
import subprocess
from glob import glob
from pprint import pprint
from idtrackerai.constants import COMPRESSED_VIDEO_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--test_names", "-t", nargs="+", type=str, default=["all"])
args = parser.parse_args()

dir_name = os.path.dirname(os.path.realpath(__file__))

all_tests = glob(os.path.join(dir_name, "test_*"))

if args.test_names != ["all"]:
    test_folders = []
    for test_name in args.test_names:
        for test_folder in all_tests:
            if test_name in test_folder:
                test_folders.append(test_folder)
else:
    test_folders = all_tests

outputs = {}
for test_folder in test_folders:
    print(test_folder)
    # We change folder to read the local_settings.py if it exists
    os.chdir(test_folder)
    command = (
        f"idtrackerai terminal_mode "
        f"--load test.json "
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
