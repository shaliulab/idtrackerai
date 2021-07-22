from typing import Tuple, Dict
import os
import json
import subprocess
import numpy as np
from idtrackerai.constants import (
    COMPRESSED_VIDEO_PATH,
    COMPRESSED_VIDEO_PATH_2,
    COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES,
    COMPRESSED_VIDEO_NUM_FRAMES,
    COMPRESSED_VIDEO_NUM_FRAMES_2,
    COMPRESSED_VIDEO_WIDTH,
    COMPRESSED_VIDEO_HEIGHT,
)
from idtrackerai.video import Video
import tempfile
from distutils.dir_util import copy_tree
import shutil
from datetime import datetime
import pytest
# Get the path to the folder where all the .json files for the tests are stored
DIR_NAME = os.path.dirname(os.path.realpath(__file__))
ASSETS_FOLDER = os.path.join(DIR_NAME, "tests_params")

# Copy the folder to a temporary folder where data will be stored
TEMP_DIR = tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S"))
assert os.path.isdir(TEMP_DIR)
copy_tree(ASSETS_FOLDER, str(TEMP_DIR))

# File tree for tests that use protocol 2
# Since there are many of them that use protocol 2, we define it as a
# global variable
DEFAULT_PROTOCOL_2_TREE = {
    "preprocessing": [
        "blobs_collection.npy",
        "fragments.npy",
        "global_fragments.npy",
        "blobs_collection_no_gaps.npy",
    ],
    "crossings_detector": [
        "supervised_crossing_detector_.checkpoint.pth",
        "supervised_crossing_detector_.checkpoint.pth",
    ],
    # there is a tracking interval so other episodes are not segmented
    "segmentation_data": [
        "episode_images_0.hdf5",
        "episode_pixels_0.hdf5",
        "episode_images_1.hdf5",
        "episode_pixels_1.hdf5",
    ],
    "identification_images": ["id_images_0.hdf5", "id_images_1.hdf5"],
    "accumulation_0": [
        "light_list_of_fragments.npy",
        "model_params.npy",
        "supervised_identification_network_.checkpoint.pth",
        "supervised_identification_network_.model.pth",
    ],
    "trajectories": ["trajectories.npy"],
    "trajectories_wo_gaps": ["trajectories_wo_gaps.npy"],
}


def _get_video_object(session_folder: str) -> Video:
    """Load the video object in a given session_folder"""
    video_object_path = os.path.join(session_folder, "video_object.npy")
    assert os.path.isfile(video_object_path)
    video_object = np.load(video_object_path, allow_pickle=True).item()
    return video_object


def _run_idtrackerai(
    root_folder: str, video_path: str = COMPRESSED_VIDEO_PATH
) -> Tuple[Dict, bool, str]:
    """Runs idtrackerai using the terminal mode

    It moves to the `root_folder` and from there executes idtrackerai on the
    video `video_path`. The `root_folder` must contain a file called
    `test.json` with the parameters used to run idtrackerai. Some test can also
    contain a file called `local_settings.py` that indicates the advanced
    parameters to be used when running idtrackerai.

    """
    # Change working directory to root_folder to read the local_settings.py
    os.chdir(root_folder)
    json_file_path = os.path.join(root_folder, "test.json")
    assert os.path.isfile(json_file_path)

    # We do not want reuse the previous a previous session folder with the
    # same name. So we make sure we delete any previous session folder with
    # the same name.

    # Get session name from test.json
    with open("test.json", "r") as f:
        input_arguments = json.load(f)
    session_name = input_arguments["_session"]["value"]

    # The session folder will be generated next to the video
    video_dir = os.path.dirname(video_path)
    original_session_folder = os.path.join(
        video_dir, f"session_{session_name}"
    )

    # Remove any session folder with the same name from potential previous
    # runs
    if os.path.isdir(original_session_folder):
        shutil.rmtree(original_session_folder)

    assert not os.path.isdir(original_session_folder)
    assert os.path.isfile(json_file_path)

    # Run idtracker.ai in terminal mode
    command = [
        "idtrackerai",
        "terminal_mode",
        "--load",
        "test.json",
        "--exec",
        "track_video",
        "--_video_path",
        f"{video_path}",
    ]
    subprocess.run(command, check=True)

    # Read Success flag from the last line of idtracker.ai logs
    with open(os.path.join(root_folder, "idtrackerai-app.log"), "r") as file:
        last_line = file.read().splitlines()[-1]

    # Store whether idtracker.ai worked as intended or not
    success_flag = False
    if "Success" in last_line:
        success_flag = True

    # We move the session folder that is next to the video in the
    # idtrackerai/data folder to the temporary folder
    moved_session_folder = os.path.join(root_folder, f"session_{session_name}")
    shutil.move(original_session_folder, moved_session_folder)

    return (
        input_arguments,
        success_flag,
        moved_session_folder,
    )


def _mandatory_outputs(session_folder):
    video_object_path = os.path.join(session_folder, "video_object.npy")
    return [video_object_path]


def _assert_input_video_object_consistency(input_arguments, session_folder):
    video_object_path = os.path.join(session_folder, "video_object.npy")
    assert os.path.isfile(video_object_path)
    video = np.load(video_object_path, allow_pickle=True).item()
    assert video.session_folder.endswith(input_arguments["_session"]["value"])
    assert (
        video.user_defined_parameters["number_of_animals"]
        == input_arguments["_number_of_animals"]["value"]
    )
    assert (
        video.user_defined_parameters["min_threshold"]
        == input_arguments["_intensity"]["value"][0]
    )
    assert (
        video.user_defined_parameters["max_threshold"]
        == input_arguments["_intensity"]["value"][1]
    )
    assert (
        video.user_defined_parameters["min_area"]
        == input_arguments["_area"]["value"][0]
    )
    assert (
        video.user_defined_parameters["max_area"]
        == input_arguments["_area"]["value"][1]
    )
    assert video.user_defined_parameters["check_segmentation"] == eval(
        input_arguments["_chcksegm"]["value"]
    )
    if input_arguments["_bgsub"] is None:
        assert video.user_defined_parameters["bkg_model"] is None
    assert input_arguments["open-multiple-files"] == video.open_multiple_files
    assert video.user_defined_parameters["track_wo_identification"] == eval(
        input_arguments["_no_ids"]["value"]
    )
    assert (
        video.user_defined_parameters["resolution_reduction"]
        == input_arguments["_resreduct"]["value"]
    )
    # TODO: assert well tracking interval for single and multiple
    # TODO: assert well apply_roi vs roi.


def _assert_files_tree(tree, session_folder, exist=True):
    for folder, files in tree.items():
        folder_path = os.path.join(session_folder, folder)
        if not exist:
            assert not os.path.isdir(folder_path)
        else:
            assert os.path.isdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not exist:
                assert not os.path.isfile(file_path)
            else:
                assert os.path.isfile(file_path)


def _assert_list_of_blobs_consistency(
    input_args, session_folder, num_frames=COMPRESSED_VIDEO_NUM_FRAMES
):
    blobs_collections = [
        "blobs_collection.npy",
        "blobs_collection_no_gaps.npy",
    ]
    # TODO: modify for the case multiple intervals
    interval = input_args["_range"]["value"]
    interval = range(interval[0], interval[1])
    for blobs_collection in blobs_collections:
        list_of_blobs_path = os.path.join(
            session_folder, "preprocessing", blobs_collection
        )
        if os.path.isfile(list_of_blobs_path):
            list_of_blobs = np.load(
                list_of_blobs_path, allow_pickle=True
            ).item()
            assert len(list_of_blobs) == num_frames
            for frame, blobs in enumerate(list_of_blobs.blobs_in_video):
                if frame in interval:
                    assert len(blobs) != 0


def _assert_background_model(session_folder):
    video_object = _get_video_object(session_folder)

    bkg_model = video_object.user_defined_parameters["bkg_model"]
    assert bkg_model is not None
    assert bkg_model.shape == (
        COMPRESSED_VIDEO_HEIGHT,
        COMPRESSED_VIDEO_WIDTH,
    )
    # background model is computed from normalized frames (divied by the mean
    # of the frame intensity).
    np.testing.assert_almost_equal(np.mean(bkg_model), 1, decimal=2)


def _assert_mask(session_folder):
    video_object = _get_video_object(session_folder)
    mask = video_object.user_defined_parameters["mask"]
    assert mask.shape == (COMPRESSED_VIDEO_HEIGHT, COMPRESSED_VIDEO_WIDTH)
    assert mask.min() == 0
    assert mask.max() == 1


def _update_local_settings_with_accumulation_folder(
    root_folder, accumulation_folder
):
    local_settings_path = os.path.join(root_folder, "local_settings.py")
    with open(local_settings_path, "r+") as file:
        content = file.read()
        file.seek(0)
        updated_content = content.replace(
            "path/to/accumulation/folder", accumulation_folder
        )
        file.write(updated_content)
        file.truncate()


# Test default run with protocol 2


@pytest.fixture(scope="module")
def default_protocol_2_run():
    root_folder = os.path.join(TEMP_DIR, "test_default_protocol_2")
    return _run_idtrackerai(root_folder)


@pytest.mark.default_protocol_2
def test_default_protocol_2_run(default_protocol_2_run):
    input_arguments, success, session_folder = default_protocol_2_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.default_protocol_2
def test_dir_tree_default_protocol_2(default_protocol_2_run):
    _, _, session_folder = default_protocol_2_run
    _assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.default_protocol_2
def test_accumulation_default_protocol2(default_protocol_2_run):
    input_arguments, _, session_folder = default_protocol_2_run
    video_object = _get_video_object(session_folder)
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video_object.ratio_accumulated_images > 0.9
    # Check that the accumulation attributes are correct
    assert video_object.accumulation_trial == 0
    assert video_object.accumulation_folder.endswith("accumulation_0")
    assert video_object.protocol1_time != 0
    assert video_object.protocol2_time != 0
    assert video_object.protocol3_pretraining_time == 0
    assert video_object.protocol3_accumulation_time == 0
    assert video_object.pretraining_folder is None


# Test resolution reduction with ROI
# Test a tracking session that enters into protocol 3
@pytest.fixture(scope="module")
def protocol3_run():
    root_folder = os.path.join(TEMP_DIR, "test_protocol3")
    return _run_idtrackerai(root_folder)


@pytest.mark.protocol3
def test_protocol3_run(protocol3_run):
    input_arguments, success, session_folder = protocol3_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.protocol3
def test_dir_tree_protocol_3(protocol3_run):
    _, _, session_folder = protocol3_run
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
            "blobs_collection_no_gaps.npy",
            "fragments.npy",
            "global_fragments.npy",
        ],
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_0.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "pretraining": [],
        "accumulation_0": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
        "trajectories": ["trajectories.npy"],
        "trajectories_wo_gaps": ["trajectories_wo_gaps.npy"],
    }
    _assert_files_tree(tree, session_folder)


@pytest.mark.xfail  # Time of protocols 1 and 2 is not correct
@pytest.mark.protocol3
def test_accumulation_protocol3(protocol3_run):
    _, _, session_folder = protocol3_run
    video_object_path = os.path.join(session_folder, "video_object.npy")
    video = np.load(video_object_path, allow_pickle=True).item()
    # The default threshold to consider protocol 2 successful is 0.9
    # see THRESHOLD_ACCEPTABLE_ACCUMULATION in constants.py
    assert video.ratio_accumulated_images < 0.9
    ratios_accumulated_images = [
        stat[-1][-1] for stat in video.accumulation_statistics
    ]
    assert video.ratio_accumulated_images == max(ratios_accumulated_images)
    best_accumulation = int(np.nanargmax(ratios_accumulated_images))
    assert video.accumulation_trial == best_accumulation
    assert video.accumulation_folder.endswith(
        f"accumulation_{best_accumulation}"
    )
    assert video.protocol1_time != 0  # TODO: protocol 1 time is not correct
    assert video.protocol2_time != 0  # TODO: protocol 2 time is not correct
    assert video.protocol3_pretraining_time != 0
    assert video.protocol3_accumulation_time != 0
    assert video.pretraining_folder is not None
    assert video.pretraining_folder.endswith("pretraining")


# Test single animal run of idtracker.ai
@pytest.fixture(scope="module")
def single_animal_run():
    root_folder = os.path.join(TEMP_DIR, "test_single_animal")
    return _run_idtrackerai(root_folder)


@pytest.mark.single_animal
def test_single_animal_run(single_animal_run):
    input_arguments, success, session_folder = single_animal_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.single_animal
def test_dir_tree_single_animal(single_animal_run):
    _, _, session_folder = single_animal_run
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
        ],
        "crossings_detector": [],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
        ],
        # Here they all appear because they are set in the video_object before
        # creating them # TODO: make this similar to segmentation
        # If no need to analyse frame do not create id_images_{}.hdf5
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "trajectories": ["trajectories.npy"],
    }
    _assert_files_tree(tree, session_folder)
    no_tree = {
        "accumulation_0": [],
        "trajectories_wo_gaps": [],
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


# Test no identities feature
@pytest.fixture(scope="module")
def wo_identification_run():
    root_folder = os.path.join(TEMP_DIR, "test_wo_identification")
    return _run_idtrackerai(root_folder)


@pytest.mark.wo_identification
def test_wo_identification(wo_identification_run):
    input_arguments, success, session_folder = wo_identification_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.wo_identification
def test_dir_tree_wo_identification(wo_identification_run):
    _, _, session_folder = wo_identification_run
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
        ],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_0.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "crossings_detector": [
            "supervised_crossing_detector_.checkpoint.pth",
            "supervised_crossing_detector_.model.pth",
        ],
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "trajectories_wo_identification": [
            "trajectories_wo_identification.npy"
        ],
    }
    _assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories": [],
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.wo_identification
def test_wo_identification_crossing_no_identified(wo_identification_run):
    _, _, session_folder = wo_identification_run
    list_of_blobs_path = os.path.join(
        session_folder, "preprocessing", "blobs_collection.npy"
    )
    list_of_blobs = np.load(list_of_blobs_path, allow_pickle=True).item()
    # Crossing are not assigned an identitiy
    assert all(
        [
            blob.identity is None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_a_crossing
        ]
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        [
            blob.identity is not None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_an_individual
        ]
    )


# Test single global fragment
@pytest.fixture(scope="module")
def single_global_fragment_run():
    root_folder = os.path.join(TEMP_DIR, "test_single_global_fragment")
    return _run_idtrackerai(root_folder)


@pytest.mark.single_global_fragment
def test_single_global_fragment(single_global_fragment_run):
    input_arguments, success, session_folder = single_global_fragment_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.single_global_fragment
def test_dir_tree_single_global_fragment(single_global_fragment_run):
    _, _, session_folder = single_global_fragment_run
    tree = {
        "preprocessing": [
            "blobs_collection.npy",
            "fragments.npy",
            "global_fragments.npy",
        ],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
        ],
        "crossings_detector": [],
        "identification_images": [
            "id_images_0.hdf5",
            "id_images_1.hdf5",
        ],
        "trajectories": ["trajectories.npy"],
    }
    _assert_files_tree(tree, session_folder)
    no_tree = {
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.single_global_fragment
def test_single_global_fragment_crossing_no_identified(
    single_global_fragment_run,
):
    _, _, session_folder = single_global_fragment_run
    list_of_blobs_path = os.path.join(
        session_folder, "preprocessing", "blobs_collection.npy"
    )
    list_of_blobs = np.load(list_of_blobs_path, allow_pickle=True).item()
    # Crossing are not assigned an identitiy
    assert all(
        [
            blob.identity is None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_a_crossing
        ]
    )
    # Individual blobs are assigned an identity but it is not a persistent
    # identity, it might change after each crossing as we are tracking
    # without identification
    assert all(
        [
            blob.identity is not None
            for blobs_in_frame in list_of_blobs.blobs_in_video
            for blob in blobs_in_frame
            if blob.is_an_individual
        ]
    )


@pytest.mark.single_global_fragment
def test_single_global_fragment_single_global_fragment(
    single_global_fragment_run,
):
    input_arguments, _, session_folder = single_global_fragment_run
    fragments_path = os.path.join(
        session_folder, "preprocessing", "fragments.npy"
    )
    list_of_fragments = np.load(fragments_path, allow_pickle=True).item()
    assert (
        len(list_of_fragments)
        == input_arguments["_number_of_animals"]["value"]
    )

    global_fragments_path = os.path.join(
        session_folder, "preprocessing", "global_fragments.npy"
    )
    list_of_global_fragments = np.load(
        global_fragments_path, allow_pickle=True
    ).item()
    assert list_of_global_fragments.number_of_global_fragments == 1


# Test a video with more blobs than number of animals where the flag
# _chcksegm is set to False
@pytest.fixture(scope="module")
def more_blobs_than_animals_chcksegm_false_run():
    root_folder = os.path.join(
        TEMP_DIR, "test_more_blobs_than_animals_chcksegm_false"
    )
    return _run_idtrackerai(root_folder)


@pytest.mark.more_blobs_than_animals_chcksegm_false
def test_more_blobs_than_animals_chcksegm_false_run(
    more_blobs_than_animals_chcksegm_false_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_false_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.more_blobs_than_animals_chcksegm_false
def test_dir_tree_more_blobs_than_animals_chcksegm_false(
    more_blobs_than_animals_chcksegm_false_run,
):
    _, _, session_folder = more_blobs_than_animals_chcksegm_false_run
    _assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.more_blobs_than_animals_chcksegm_false
def test_more_blobs_than_animals_chcksegm_false_more_blobs_than_animals(
    more_blobs_than_animals_chcksegm_false_run,
):
    (
        input_arguments,
        _,
        session_folder,
    ) = more_blobs_than_animals_chcksegm_false_run
    list_of_blobs_path = os.path.join(
        session_folder, "preprocessing", "blobs_collection.npy"
    )
    number_of_animals = input_arguments["_number_of_animals"]["value"]
    list_of_blobs = np.load(list_of_blobs_path, allow_pickle=True).item()
    assert any(
        [
            len(blobs_in_frame) > number_of_animals
            for blobs_in_frame in list_of_blobs.blobs_in_video
        ]
    )


# Forcing background subtraction to use the mean statistic creates
# more blobs than animals in some frames
# Test a segmentation with more blobs than number of animals where the flag
# _chcksegm is set to True
@pytest.fixture(scope="module")
def background_subtraction_mean_run():
    root_folder = os.path.join(TEMP_DIR, "test_bkg_subtraction_mean")
    return _run_idtrackerai(root_folder)


@pytest.mark.background_subtraction_mean
def test_bkg_subtraction_mean_run(
    background_subtraction_mean_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_mean_run
    # Tracking does not return a positive success flag because it is
    # intended to fail when the maximum number of blobs is greater than the
    # number of animals indicated in the input arguments and the chcksegm flag
    # is set to True.
    assert not success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)
    inconsistent_frames_path = os.path.join(
        session_folder, "inconsistent_frames.csv"
    )
    assert os.path.exists(inconsistent_frames_path)


@pytest.mark.background_subtraction_mean
def test_dir_tree_background_subtraction_mean_run(
    background_subtraction_mean_run,
):
    _, _, session_folder = background_subtraction_mean_run
    tree = {
        "preprocessing": ["blobs_collection.npy"],
        # there is a tracking interval so other episodes are not segmented
        "segmentation_data": [
            "episode_images_0.hdf5",
            "episode_pixels_0.hdf5",
            "episode_images_1.hdf5",
            "episode_pixels_1.hdf5",
        ],
        "identification_images": [],
    }
    _assert_files_tree(tree, session_folder)
    no_tree = {
        "crossings_detector": [],
        "trajectories": [],
        "trajectories_wo_gaps": [],
        "accumulation_0": [],
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.background_subtraction_mean
def test_background_subtraction_mean_bkg_model(
    background_subtraction_mean_run,
):
    _, _, session_folder = background_subtraction_mean_run
    _assert_background_model(session_folder)


# Test tracking a video using background subtraction
# (default uses min statistic)
@pytest.fixture(scope="module")
def background_subtraction_run():
    root_folder = os.path.join(TEMP_DIR, "test_bkg_subtraction_default")
    return _run_idtrackerai(root_folder)


@pytest.mark.background_subtraction_default
def test_background_subtraction_run(background_subtraction_run):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.background_subtraction_default
def test_dir_tree_background_subtraction(
    background_subtraction_run,
):
    _, _, session_folder = background_subtraction_run
    _assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.background_subtraction_default
def test_background_subtraction_default_bkg_model(background_subtraction_run):
    _, _, session_folder = background_subtraction_run
    _assert_background_model(session_folder)


# Test ROI with BKG
@pytest.fixture(scope="module")
def background_subtraction_with_ROI_run():
    root_folder = os.path.join(TEMP_DIR, "test_bkg_roi")
    return _run_idtrackerai(root_folder)


@pytest.mark.background_subtraction_with_ROI
def test_background_subtraction_with_ROI_run(
    background_subtraction_with_ROI_run,
):
    (
        input_arguments,
        success,
        session_folder,
    ) = background_subtraction_with_ROI_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(input_arguments, session_folder)


@pytest.mark.background_subtraction_with_ROI
def test_dir_tree_background_subtraction(
    background_subtraction_with_ROI_run,
):
    _, _, session_folder = background_subtraction_with_ROI_run
    _assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


@pytest.mark.background_subtraction_with_ROI
def test_background_subtraction_with_ROI_bkg_model(
    background_subtraction_with_ROI_run,
):
    _, _, session_folder = background_subtraction_with_ROI_run
    _assert_background_model(session_folder)


# Test multiple files
@pytest.fixture(scope="module")
def multiple_files_run():
    root_folder = os.path.join(TEMP_DIR, "test_multiple_files")
    return _run_idtrackerai(root_folder)


@pytest.mark.multiple_files
def test_multiple_files_run(
    multiple_files_run,
):
    input_arguments, success, session_folder = multiple_files_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_MULTIPLE_FILES,
    )


@pytest.mark.multiple_files
def test_dir_tree_multiple_files(
    multiple_files_run,
):
    _, _, session_folder = multiple_files_run
    _assert_files_tree(DEFAULT_PROTOCOL_2_TREE, session_folder)
    no_tree = {
        "pretraining": [],
        "accumulation_1": [],
        "accumulation_2": [],
        "accumulation_3": [],
    }
    _assert_files_tree(no_tree, session_folder, exist=False)


# Test knowledge transfer
@pytest.fixture(scope="module")
def knowledge_transfer_run(default_protocol_2_run):
    _, _, session_folder = default_protocol_2_run
    accumulation_folder = os.path.join(session_folder, "accumulation_0")
    root_folder = os.path.join(TEMP_DIR, "test_knowledge_transfer")
    _update_local_settings_with_accumulation_folder(
        root_folder, accumulation_folder
    )
    return _run_idtrackerai(root_folder, video_path=COMPRESSED_VIDEO_PATH_2)


@pytest.mark.knowledge_transfer
def test_knowledge_transfer_run(knowledge_transfer_run):
    input_arguments, success, session_folder = knowledge_transfer_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2,
    )


@pytest.mark.knowledge_transfer
def test_knowledge_transfer_happened(knowledge_transfer_run):
    _, _, session_folder = knowledge_transfer_run
    video_object = _get_video_object(session_folder)
    assert video_object.user_defined_parameters["knowledge_transfer_folder"]

    root_folder = os.path.dirname(session_folder)
    log_file_path = os.path.join(root_folder, "idtrackerai-app.log")
    with open(log_file_path, "r") as log_file:
        logs = log_file.read()
        assert "Tracking with knowledge transfer" in logs
        assert "Reinitializing fully connected layers" in logs


# Test identity transfer
# This also tests protocol 1
@pytest.fixture(scope="module")
def identity_transfer_run(default_protocol_2_run):
    _, _, session_folder = default_protocol_2_run
    accumulation_folder = os.path.join(session_folder, "accumulation_0")
    root_folder = os.path.join(TEMP_DIR, "test_identity_transfer")
    _update_local_settings_with_accumulation_folder(
        root_folder, accumulation_folder
    )
    return _run_idtrackerai(root_folder, video_path=COMPRESSED_VIDEO_PATH_2)


@pytest.mark.identity_transfer
def test_identity_transfer_run(identity_transfer_run):
    input_arguments, success, session_folder = identity_transfer_run
    assert success
    _assert_input_video_object_consistency(input_arguments, session_folder)
    _assert_list_of_blobs_consistency(
        input_arguments,
        session_folder,
        num_frames=COMPRESSED_VIDEO_NUM_FRAMES_2,
    )


@pytest.mark.identity_transfer
def test_identity_transfer_happened(identity_transfer_run):
    _, _, session_folder = identity_transfer_run
    video_object = _get_video_object(session_folder)
    assert video_object.user_defined_parameters["knowledge_transfer_folder"]
    assert video_object.user_defined_parameters["identity_transfer"]
    # TODO: This is not truly a user defined parameter
    assert video_object.user_defined_parameters[
        "identification_image_size"
    ] == (42, 42, 1)

    kt_folder = video_object.user_defined_parameters[
        "knowledge_transfer_folder"
    ]
    root_folder = os.path.dirname(session_folder)
    log_file_path = os.path.join(root_folder, "idtrackerai-app.log")
    with open(log_file_path, "r") as log_file:
        logs = log_file.read()
        assert "Tracking with knowledge transfer" in logs
        assert (
            "Identity transfer. Not reinitializing the fully connected layers."
            in logs
        )
        assert "Identities transferred successfully" in logs
        assert f"Transferring identities from {kt_folder}" in logs
        assert "Protocol 1 successful" in logs


# TODO: Code test max_number_of_blobs < number_of_animals
# TODO: Code test save pixels
# TODO: Code test save segmentation images
# TODO: Code test data policy
# TODO: Code test save CSV data
# TODO: Code test lower MAX_RATIO_OF_PRETRAINED_IMAGES
# TODO: Code test sigma blurring

# if True:
#     shutil.rmtree(TEMP_DIR)
