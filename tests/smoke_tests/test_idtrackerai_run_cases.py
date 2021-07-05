import os
import pytest
import json
import subprocess
from idtrackerai.constants import COMPRESSED_VIDEO_PATH
import tempfile
from distutils.dir_util import copy_tree
import shutil

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
ASSETS_FOLDER = os.path.join(DIR_NAME, "assets")


# Copy assets to a temporary folder where data will be stored
TEMP_DIR = tempfile.mkdtemp()
copy_tree(ASSETS_FOLDER, str(TEMP_DIR))

print(TEMP_DIR)


def run_idtrackerai(root_folder):
    """Runs idtrackerai in terminal_mode from the root_folder"""
    # Change working directory to root_folder to read the local_settings.py
    os.chdir(root_folder)
    json_file_path = os.path.join(root_folder, "test.json")

    assert os.path.isfile(json_file_path)

    command = [
        "idtrackerai",
        "terminal_mode",
        "--load",
        "test.json",
        "--exec",
        "track_video",
        "--_video_path",
        f"{COMPRESSED_VIDEO_PATH}",
    ]
    subprocess.run(command, check=True)

    # Read Success flag from the last line of idtracker.ai logs
    with open(os.path.join(root_folder, "idtrackerai-app.log"), "r") as file:
        last_line = file.read().splitlines()[-1]

    # Store idtracker.ai worked or not
    success_flag = False
    if "Success" in last_line:
        success_flag = True

    # Get session name from test.json
    with open("test.json", "r") as f:
        input_arguments = json.load(f)
    session_name = input_arguments["_session"]["value"]
    video_dir = os.path.dirname(COMPRESSED_VIDEO_PATH)
    original_session_folder = os.path.join(
        video_dir, f"session_{session_name}"
    )
    moved_session_folder = os.path.join(root_folder, f"session_{session_name}")
    shutil.move(original_session_folder, moved_session_folder)

    return (
        input_arguments,
        success_flag,
        moved_session_folder,
    )


def mandatory_outputs(session_folder):
    video_object_path = os.path.join(session_folder, "video_object.npy")
    return [video_object_path]


# Test default run of idtracker.ai
@pytest.fixture(scope="module")
def default_protocol_2_run():
    root_folder = os.path.join(TEMP_DIR, "test_default_protocol_2")
    return run_idtrackerai(root_folder)


@pytest.mark.default_protocol_2
def test_default_protocol_2_output(default_protocol_2_run):
    input_arguments, success, session_folder = default_protocol_2_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of default run


@pytest.mark.default_protocol_2
def test_other_things_default_run(default_protocol_2_run):
    input_arguments, success, session_folder = default_protocol_2_run
    assert success


# Test a tracking session that enters into protocol 3
@pytest.fixture(scope="module")
def protocol3_run():
    root_folder = os.path.join(TEMP_DIR, "test_protocol3")
    return run_idtrackerai(root_folder)


@pytest.mark.protocol3
def test_protocol3_output(protocol3_run):
    input_arguments, success, session_folder = protocol3_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of protocol3


# Test single animal run of idtracker.ai
@pytest.fixture(scope="module")
def single_animal_run():
    root_folder = os.path.join(TEMP_DIR, "test_single_animal")
    return run_idtrackerai(root_folder)


@pytest.mark.single_animal
def test_output_single_animal(single_animal_run):
    input_arguments, success, session_folder = single_animal_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of single_animal run


# Test no identities feature
@pytest.fixture(scope="module")
def wo_identities_run():
    root_folder = os.path.join(TEMP_DIR, "test_wo_identities")
    return run_idtrackerai(root_folder)


@pytest.mark.wo_identities
def test_output_wo_identities(wo_identities_run):
    input_arguments, success, session_folder = wo_identities_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of wo_identities run


# Test single global fragment
@pytest.fixture(scope="module")
def single_global_fragment_run():
    root_folder = os.path.join(TEMP_DIR, "test_single_global_fragment")
    return run_idtrackerai(root_folder)


@pytest.mark.single_global_fragment
def test_output_single_global_fragment(single_global_fragment_run):
    input_arguments, success, session_folder = single_global_fragment_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of wo_identities run


# Test a video with more blobs than number of animals where the flag
# _chcksegm is set to False
@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_false_run():
    root_folder = os.path.join(
        TEMP_DIR, "test_more_blobs_than_animals_chcksegm_false"
    )
    return run_idtrackerai(root_folder)


@pytest.mark.more_blobs_than_animals_chcksegm_false
def test_more_blobs_than_animals_chcksegm_false_output(
    more_blobs_than_animals_chcksegm_false_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_false_run
    assert success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)

    # TODO: test particular outputs of more_blobs_than_animals_chcksegm_false


# Test a video with more blobs than number of animals where the flag
# _chcksegm is set to True
@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_true_run():
    root_folder = os.path.join(
        TEMP_DIR, "test_more_blobs_than_animals_chcksegm_true"
    )
    return run_idtrackerai(root_folder)


# This test fails becuase idtracker.ai raises an exception when the
# maximum number of blobs is greater than the number of animals in the video
# and the _chcksegm flag is set to True.
@pytest.mark.more_blobs_than_animals_chcksegm_true
def test_more_blobs_than_animals_chcksegm_true_output(
    more_blobs_than_animals_chcksegm_true_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_true_run
    assert not success
    for path in mandatory_outputs(session_folder):
        assert os.path.exists(path)
    inconsistent_frames_path = os.path.join(
        session_folder, "inconsistent_frames.csv"
    )
    assert os.path.exists(inconsistent_frames_path)

    # TODO: test particular outputs of more_blobs_than_animals_chcksegm_true


# TODO: Code test for bkg (min, max, mean)
# TODO: Code test for ROI
# TODO: Code test for ROI with resolution reduction
# TODO: COde test for ROI with bkg
# TODO: Code test for multiple files
# TODO: Code test for knowledge transfer
# TODO: Code test for identity transfer
# TODO: Code test max_number_of_blobs < number_of_animals
# TODO: Code test intervals (already in single global fragment and single animal)
# TODO: Code test resolution reduction (already in protocol 3)
# TODO: Code test save pixels
# TODO: Code test save segmentation images
# TODO: Code test

# shutil.rmtree(TEMP_DIR)
