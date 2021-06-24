# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H.,
# de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of
# unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P:
# gonzalo.polavieja@neuro.fchampalimaud.org)

import logging
import os
from pprint import pprint

import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs

"""Given two list of blobs_in_video, one deduced from human ground_truth and the 
other generated by the tracking algorithm, compares them and gives back some
statistics

Crossing: crossings are a special case. We ...
"""


logger = logging.getLogger("__main__.compute_statistics_against_groundtruth")


def get_corresponding_gt_blob(blob, gt_blobs_in_frame):
    """Returs the blobs in the `gt_blobs_in_frame` that overlap (in pixels)
    with a given `blob` of the tracked trajectories.

    :param blob: <Blob object>
    :param gt_blobs_in_frame: list of <GroundTruthBlob objects> corresponding
    to the same frame
    :return: list of <GroundTruthBlob objects> that overlap in pixels with
    `blob`
    """
    blobs_of_same_frame = [
        blob.frame_number == gt_blob.frame_number
        for gt_blob in gt_blobs_in_frame
    ]
    assert all(blobs_of_same_frame)
    corresponding_gt_blobs = []
    for gt_blob in gt_blobs_in_frame:
        if set(gt_blob.pixels).intersection(set(blob.pixels)):
            corresponding_gt_blobs.append(gt_blob)
    return corresponding_gt_blobs


def update_sum_indiv_P2(gt_id, blob, results):
    identified = blob.assigned_identities[0] != 0
    # we only consider P2 for non interpolated blobs
    id_before_interp = blob.identities_corrected_closing_gaps is None
    if identified and id_before_interp:
        results["sum_indiv_P2"][gt_id] += blob._P2_vector[gt_id - 1]


def update_results_with_id_error(results, blob, gt_id):
    results["errors_blobs"][gt_id] += 1
    results["frames_w_id_errors"] = results["frames_w_id_errors"].union(
        {blob.frame_number}
    )
    if blob.fragment_identifier:
        results["frag_w_id_errors"] = results["frag_w_id_errors"].union(
            {blob.fragment_identifier}
        )

    if not blob.used_for_training:
        results["errors_blobs_after_accum"][gt_id] += 1

    # Errors of identified blobs
    if blob.assigned_identities[0] != 0:
        results["errors_id_blobs"][gt_id] += 1

        if blob.used_for_training:
            results["errors_id_blobs_accum"][gt_id] += 1
        else:
            results["errors_id_blobs_after_accum"][gt_id] += 1


def update_results_for_identified_gt_blob(results, blob, gt_id):
    update_sum_indiv_P2(gt_id, blob, results)

    # Count blobs
    results["num_blobs"][gt_id] += 1
    if blob.used_for_training:
        results["num_id_blobs_accum"][gt_id] += 1
    else:
        results["num_blobs_after_accum"][gt_id] += 1

    if blob.assigned_identities[0] != 0:
        results["num_id_blobs"][gt_id] += 1

    if gt_id != blob.assigned_identities[0]:
        update_results_with_id_error(results, blob, gt_id)


def compare_blob_with_gt_blob(results, blob, gt_blob, ids_perm_dict):
    if ids_perm_dict is not None:
        gt_id = ids_perm_dict[gt_blob.gt_identity]
    else:
        gt_id = gt_blob.gt_identity

    if gt_id == 0:
        # This is here to raise and error at the end of the computations
        # A ground truth individual blob cannot have identity 0.
        results["frames_w_0_id_in_gt"] = results["frames_w_0_id_in_gt"].union(
            {gt_blob.frame_number}
        )
    else:
        update_results_for_identified_gt_blob(results, blob, gt_id)


def compare_frame(results, blobs_in_frame, gt_blobs_in_frame, ids_perm_dict):
    for blob in blobs_in_frame:
        corresponding_gt_blobs = get_corresponding_gt_blob(
            blob, gt_blobs_in_frame
        )
        if len(corresponding_gt_blobs) == 1:
            gt_blob = corresponding_gt_blobs[0]
            cond1 = gt_blob.is_an_individual
            cond2 = gt_blob.gt_identity != -1
            cond3 = not gt_blob.was_a_crossing
            gt_blob_is_individual = cond1 and cond2 and cond3
            if blob.is_an_individual and gt_blob_is_individual:
                results["num_indiv_gt_blobs"] += 1
                results["num_indiv_blobs"] += 1
                results["crossing_detector_tn"] += 1
                compare_blob_with_gt_blob(
                    results, blob, gt_blob, ids_perm_dict
                )
            elif blob.is_an_individual and not gt_blob_is_individual:
                # ground truth crossing blob. This could mean that the new
                # video has a better segmentation and it would not be a
                # crossing detection error.
                results["num_crossing_gt_blobs"] += 1
                results["num_indiv_blobs"] += 1
                results["crossing_detector_fn"] += 1
                results["frame_with_crossing_detection_error"] = results[
                    "frame_with_crossing_detection_error"
                ].union({blob.frame_number})
            elif blob.is_a_crossing and gt_blob_is_individual:
                results["num_indiv_gt_blobs"] += 1
                results["num_crossing_blobs"] += 1
                results["crossing_detector_fp"] += 1
                results["frame_with_crossing_detection_error"] = results[
                    "frame_with_crossing_detection_error"
                ].union({blob.frame_number})
            elif blob.is_a_crossing and not gt_blob_is_individual:
                results["num_crossing_gt_blobs"] += 1
                results["num_crossing_blobs"] += 1
                results["crossing_detector_tp"] += 1
        else:
            if blob.is_an_individual:
                results["num_indiv_blobs"] += 1
            else:
                results["num_crossing_blobs"] += 1

    return results


def per_id_counter_dict(num_animals):
    return {i: 0 for i in range(1, num_animals + 1)}


def init_results_dict(num_animals):
    results = {}
    results["num_blobs"] = per_id_counter_dict(num_animals)
    results["sum_indiv_P2"] = per_id_counter_dict(num_animals)
    results["num_id_blobs"] = per_id_counter_dict(num_animals)
    results["num_id_blobs_accum"] = per_id_counter_dict(num_animals)
    results["num_blobs_after_accum"] = per_id_counter_dict(num_animals)
    results["errors_blobs"] = per_id_counter_dict(num_animals)
    results["errors_id_blobs"] = per_id_counter_dict(num_animals)
    results["errors_blobs_after_accum"] = per_id_counter_dict(num_animals)
    results["errors_id_blobs_accum"] = per_id_counter_dict(num_animals)
    results["errors_id_blobs_after_accum"] = per_id_counter_dict(num_animals)

    results["frames_w_id_errors"] = set()
    results["frag_w_id_errors"] = set()
    results["frames_w_crossing_errors"] = set()
    results["frag_w_crossing_errors"] = set()
    results["frames_w_0_id_in_gt"] = set()
    results["num_crossing_frags"] = 0
    results["frag_crossings"] = set()

    # Crossing detector
    results["crossing_detector_tn"] = 0
    results["crossing_detector_fn"] = 0
    results["crossing_detector_tp"] = 0
    results["crossing_detector_fp"] = 0

    results["num_indiv_gt_blobs"] = 0
    results["num_crossing_gt_blobs"] = 0
    results["num_indiv_blobs"] = 0
    results["num_crossing_blobs"] = 0

    results["frame_with_crossing_detection_error"] = set()

    return results


def aggregate_counters(results):
    results["total_sum_P2"] = np.sum(list(results["sum_indiv_P2"].values()))
    results["total_num_blobs"] = (
        results["num_indiv_gt_blobs"] + results["num_crossing_gt_blobs"]
    )
    results["total_indiv_blobs"] = np.sum(list(results["num_blobs"].values()))
    results["total_num_errors"] = np.sum(
        list(results["errors_blobs"].values())
    )
    results["total_assigned_blobs"] = np.sum(
        list(results["num_id_blobs"].values())
    )
    results["total_errors_assigned_blobs"] = np.sum(
        list(results["errors_id_blobs"].values())
    )
    results["total_id_blobs_accum"] = np.sum(
        list(results["num_id_blobs_accum"].values())
    )
    results["total_errors_accum"] = np.sum(
        list(results["errors_id_blobs_accum"].values())
    )
    results["total_id_blobs_after_accum"] = np.sum(
        list(results["num_blobs_after_accum"].values())
    )
    results["total_errors_after_accum"] = np.sum(
        list(results["errors_blobs_after_accum"].values())
    )


def compare_tracking_with_ground_truth(
    num_animals,
    gt_blobs_in_video,
    blobs_in_video,
    ids_perm_dict,
):
    """
    This function only considers individual blobs

    :param num_animals:
    :param gt_blobs_in_video:
    :param blobs_in_video:
    :param ids_perm_dict:
    :return:
    """
    # create dictionary to store counters
    results = init_results_dict(num_animals)
    both_blobs_in_video = zip(gt_blobs_in_video, blobs_in_video)
    for gt_blobs_in_frame, blobs_in_frame in both_blobs_in_video:
        results = compare_frame(
            results, blobs_in_frame, gt_blobs_in_frame, ids_perm_dict
        )
    aggregate_counters(results)
    return results


def check_gt_video_consistency(video, gt_video):
    """Checks that the `video` and `gt_video` are consistent: they have
    the same number of frames, the same number animals.

    :param video: <Video object> of the tracked video
    :param gt_video: <Video object> of the tracking session from which the
    ground_truth file was computed
    """
    if video.number_of_frames != gt_video.number_of_frames:
        raise ValueError(
            "The number of frames in the video and in the ground truth video "
            "are different. The ground truth file cannot be reused"
        )

    if (
        video.user_defined_parameters["number_of_animals"]
        != gt_video.user_defined_parameters["number_of_animals"]
    ):
        raise ValueError(
            "The number of animals in the video and in the ground truth video "
            "are different. The ground truth file cannot be reused"
        )

    video_frame_dims = (video.original_height, video.original_width)
    gt_video_frame_dims = (gt_video.original_height, gt_video.original_width)
    if video_frame_dims != gt_video_frame_dims:
        raise ValueError(
            "The video frame dimensions of the ground truth video"
            "and the new video are different"
        )


def get_ids_perm_dict(gt_blobs_in_frame, blobs_in_frame):
    ids_perm_dict = {}
    for blob in blobs_in_frame:
        corresponding_blobs = get_corresponding_gt_blob(
            blob, gt_blobs_in_frame
        )
        if len(corresponding_blobs) == 1:
            ids_perm_dict[
                corresponding_blobs[0].gt_identity
            ] = blob.assigned_identities[0]
        else:
            break
    return ids_perm_dict


def get_permutation_of_identities(
    video,
    fff_global_fragment,
    gt_blobs_in_video,
    blobs_in_video,
):
    """Returns a dictionary with the permutation of identities to be
    considered when comparing identities of the ground truth data with the
    new trajectories

    :param video: <Video object> of the new video
    :param fff_global_fragment: int indicating the first frame
    of the core of the first global fragment of the new video
    :param gt_blobs_in_video: list of lists of <GroundTruthBlob
    object> with the ground truth blobs of each frame
    :param blobs_in_video: list of lists of <Blob object> with the blobs
    of each frame of the new video
    :return: dict, key is identity in gt data, value is identity in new video
    """
    ids_perm_dict = {}
    permutation_found = False
    while not permutation_found:

        gt_blobs_in_frame = gt_blobs_in_video[fff_global_fragment]
        blobs_in_frame = blobs_in_video[fff_global_fragment]

        if (
            len(gt_blobs_in_frame)
            == video.user_defined_parameters["number_of_animals"]
        ):
            ids_perm_dict = get_ids_perm_dict(
                gt_blobs_in_frame, blobs_in_frame
            )
            if (
                len(ids_perm_dict)
                == video.user_defined_parameters["number_of_animals"]
            ):
                permutation_found = True
            else:
                fff_global_fragment += 1
        else:
            fff_global_fragment += 1

        if fff_global_fragment > video.number_of_frames:
            raise Exception("No identities permutation found")
    logger.info(f"The identities permutation is {ids_perm_dict}")
    assert (
        len(ids_perm_dict)
        == video.user_defined_parameters["number_of_animals"]
    )
    return ids_perm_dict


def compute_performance(results, number_of_animals):
    accuracies = {}

    accuracies["percentage_of_unoccluded_images"] = (
        results["num_indiv_gt_blobs"] / results["total_num_blobs"]
    )

    accuracies["mean_individual_P2_in_validated_part"] = (
        results["total_sum_P2"] / results["total_indiv_blobs"]
    )

    error_rate = results["total_num_errors"] / results["total_indiv_blobs"]
    accuracies["accuracy"] = 1.0 - error_rate

    error_rate = (
        results["total_errors_assigned_blobs"]
        / results["total_assigned_blobs"]
    )
    accuracies["accuracy_assigned"] = 1.0 - error_rate

    error_rate = (
        results["total_errors_accum"] / results["total_id_blobs_accum"]
    )
    accuracies["accuracy_in_accumulation"] = 1.0 - error_rate

    if results["total_id_blobs_after_accum"] != 0:
        error_rate = (
            results["total_errors_after_accum"]
            / results["total_id_blobs_after_accum"]
        )
        accuracies["accuracy_after_accumulation"] = 1.0 - error_rate
    else:
        accuracies["accuracy_after_accumulation"] = None

    if results["num_crossing_gt_blobs"] != 0:
        correct = (
            results["crossing_detector_tn"] + results["crossing_detector_tp"]
        )
        positive = (
            results["crossing_detector_tp"] + results["crossing_detector_fp"]
        )
        negative = (
            results["crossing_detector_tn"] + results["crossing_detector_fn"]
        )
        total = positive + negative
        accuracies["crossing_detector_accuracy"] = correct / total
        accuracies["crossing_detector_precision"] = (
            results["crossing_detector_tp"] / positive
        )
        accuracies["crossing_detector_recall"] = results[
            "crossing_detector_tp"
        ] / (results["crossing_detector_tp"] + results["crossing_detector_fn"])
    else:
        accuracies["crossing_detector_accuracy"] = None

    accuracies["individual_P2_in_validated_part"] = {}
    accuracies["individual_accuracy"] = {}
    accuracies["individual_accuracy_assigned"] = {}
    accuracies["individual_accuracy_in_accumulation"] = {}
    accuracies["individual_accuracy_after_accumulation"] = {}
    for i in range(1, number_of_animals + 1):
        accuracies["individual_P2_in_validated_part"][i] = (
            results["sum_indiv_P2"][i] / results["num_blobs"][i]
        )
        if results["num_blobs"] != 0:
            error_rate = results["errors_blobs"][i] / results["num_blobs"][i]
            accuracies["individual_accuracy"][i] = 1 - error_rate
        else:
            accuracies["individual_accuracy"][i] = None

        if results["num_id_blobs"] != 0:
            error_rate = (
                results["errors_id_blobs"][i] / results["num_id_blobs"][i]
            )
            accuracies["individual_accuracy_assigned"][i] = 1 - error_rate
        else:
            accuracies["individual_accuracy_assigned"][i] = None

        if results["num_id_blobs_accum"][i] != 0:
            error_rate = (
                results["errors_id_blobs_accum"][i]
                / results["num_id_blobs_accum"][i]
            )
            accuracies["individual_accuracy_in_accumulation"][i] = (
                1 - error_rate
            )
        else:
            accuracies["individual_accuracy_in_accumulation"][i] = None

        if results["num_blobs_after_accum"][i] != 0:
            error_rate = (
                results["errors_blobs_after_accum"][i]
                / results["num_blobs_after_accum"][i]
            )
            accuracies["individual_accuracy_after_accumulation"][i] = (
                1 - error_rate
            )
        else:
            accuracies["individual_accuracy_after_accumulation"][i] = None

    logger.info("accuracies %s" % str(accuracies))
    return accuracies


def get_accuracy_wrt_groundtruth(
    video,
    gt_blobs_in_video,
    blobs_in_video=None,
    identities_dictionary_permutation=None,
):
    number_of_animals = video.user_defined_parameters["number_of_animals"]
    if blobs_in_video is None:
        blobs_in_video = gt_blobs_in_video

    results = compare_tracking_with_ground_truth(
        number_of_animals,
        gt_blobs_in_video,
        blobs_in_video,
        identities_dictionary_permutation,
    )
    pprint(results)

    if len(results["frames_w_0_id_in_gt"]) == 0:
        accuracies = compute_performance(results, number_of_animals)
        return accuracies, results
    else:
        logger.info(
            "there are fish with 0 identity in frame %s"
            % str(results["frames_w_0_id_in_gt"])
        )
        return None, results


def reduce_pixels(
    blob, original_width, original_height, width, height, resolution_reduction
):
    pxs = np.array(
        np.unravel_index(blob.pixels, (original_height, original_width))
    ).T
    pxs_reduced = (np.round(pxs * resolution_reduction)).astype("int")
    pxs_reduced = np.ravel_multi_index(
        [pxs_reduced[:, 0], pxs_reduced[:, 1]], (height, width)
    )
    return pxs_reduced


def reduce_resolution_gt_blobs(video, gt_blobs_in_video):
    for gt_blobs_in_frame in gt_blobs_in_video:
        for gt_blob in gt_blobs_in_frame:
            gt_blob.pixels = reduce_pixels(
                gt_blob,
                video.original_width,
                video.original_height,
                video.width,
                video.height,
                video.user_defined_parameters["resolution_reduction"],
            )


def compute_and_save_session_accuracy_wrt_groundtruth(video, gt_type=None):

    if gt_type == "normal":
        list_of_blobs_path = video.blobs_path
        gt_path = os.path.join(video.video_folder, "_groundtruth.npy")
        performance_func = get_accuracy_wrt_groundtruth
    elif gt_type == "no_gaps":
        raise Exception(f"No performance_func to compute for {gt_type}")
        ### TODO: fixh get_accuracy_wrt_groundtruth_no_gaps
        # list_of_blobs_path = video.blobs_no_gaps_path
        # gt_path = os.path.join(
        #     video.video_folder, "_groundtruth_with_crossing_identified.npy"
        # )
        # performance_func = get_accuracy_wrt_groundtruth_no_gaps

    else:
        raise ValueError(f"Not valid gt_type {gt_type}")

    logger.info("loading list_of_blobs")
    list_of_blobs = ListOfBlobs.load(list_of_blobs_path)

    logger.info("loading ground truth")
    ground_truth = np.load(
        gt_path, allow_pickle=True, encoding="latin1"
    ).item()

    check_gt_video_consistency(video, ground_truth.video)

    if video.user_defined_parameters["resolution_reduction"] != 1:
        reduce_resolution_gt_blobs(video, ground_truth.blobs_in_video)

    accumulation_number = int(video.accumulation_folder[-1])
    identities_dictionary_permutation = get_permutation_of_identities(
        video,
        video.first_frame_first_global_fragment[accumulation_number],
        ground_truth.blobs_in_video,
        list_of_blobs.blobs_in_video,
    )

    # Select the frame for which we checked the identities and computed the
    # ground_truth file
    gt_blobs_in_video = ground_truth.blobs_in_video[
        ground_truth.start : ground_truth.end
    ]

    blobs_in_video = list_of_blobs.blobs_in_video[
        ground_truth.start : ground_truth.end
    ]

    logger.info("computing performance")
    accuracies, results = performance_func(
        video,
        gt_blobs_in_video,
        blobs_in_video,
        identities_dictionary_permutation,
    )

    if accuracies is not None:
        save_accuracies_in_video(
            video,
            accuracies,
            results,
            (ground_truth.start, ground_truth.end),
            gt_type,
        )

    return video, ground_truth


def save_accuracies_in_video(
    video, accuracies, results, gt_start_end, gt_type
):
    logger.info("saving accuracies in video")
    video.gt_start_end = gt_start_end
    if gt_type == "normal":
        video.gt_accuracy = accuracies
        video.gt_results = results
    elif gt_type == "no_gaps":
        video.gt_accuracy_no_gaps = accuracies
        video.gt_results_no_gaps = results
    video.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gt",
        "--gt_type",
        type=str,
        help="type of ground_truth to compute \
                        ('no_gaps' or 'normal')",
    )
    parser.add_argument(
        "-sf", "--session_folder", type=str, help="path to the session folder"
    )
    args = parser.parse_args()

    gt_type = args.gt_type
    session_folder = args.session_folder
    video_object_path = os.path.join(session_folder, "video_object.npy")
    logger.info("loading video object")
    video = np.load(
        video_object_path, allow_pickle=True, encoding="latin1"
    ).item(0)
    video.update_paths(video_object_path)
    gt_path = os.path.join(video.video_folder, "_groundtruth.npy")
    ground_truth = np.load(
        gt_path, allow_pickle=True, encoding="latin1"
    ).item()
    video, ground_truth = compute_and_save_session_accuracy_wrt_groundtruth(
        video, gt_type
    )
    print(video.gt_accuracy)
