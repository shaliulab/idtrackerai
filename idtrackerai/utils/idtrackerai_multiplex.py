"""
Functionality to quickly check the accuracy of many sessions
"""
import argparse
import logging
import re

from idtrackerai.utils.py_utils import is_idtrackerai_folder

logger = logging.getLogger(__name__)


def list_accuracy(folders):

    accuracies = {}
    for session_folder in folders:
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
                    accuracy = float(match.group(1))

        accuracies[session_folder] = accuracy

    return accuracies


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
    return ap


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        ap.parse_args()

    args = ap.parse_args()

    folders = validate_sessions(args.interval)
    accuracies = list_accuracy(folders)
    for sf, acc in accuracies.items():
        print(f"{sf}: {round(acc, 3)}")


if __name__ == "__main__":
    main()
