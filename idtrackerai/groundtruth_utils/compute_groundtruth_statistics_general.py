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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys
import numpy as np
from pprint import pprint
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.blob import Blob
from idtrackerai.groundtruth_utils.generate_groundtruth import (
    GroundTruth,
    GroundTruthBlob,
)

"""Given two list of blobs_in_video, one deduced from human groundtruth and the other
generated by the tracking algorithm, compares them and gives back some statistics

Crossing: crossings are a special case. We ...
"""

import logging

logger = logging.getLogger("__main__.compute_statistics_against_groundtruth")


def get_corresponding_gt_blob(blob, gt_blobs_in_frame):
    corresponding_gt_blobs = []
    for gt_blob in gt_blobs_in_frame:
        num_overlapping_pixels = len(set(gt_blob.pixels) & set(blob.pixels))
        if num_overlapping_pixels > 0:
            corresponding_gt_blobs.append(gt_blob)
    return corresponding_gt_blobs


def compare_tracking_against_groundtruth(
    number_of_animals,
    blobs_in_video_groundtruth,
    blobs_in_video,
    identities_dictionary_permutation,
):
    # create dictionary to store eventual corrections made by the user
    results = {}
    results["number_of_blobs_per_identity"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["sum_individual_P2"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_assigned_blobs_per_identity"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_blobs_assigned_during_accumulation_per_identity"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_blobs_after_accumulation_per_identity"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_errors_in_all_blobs"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_errors_in_assigned_blobs"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_errors_in_blobs_assigned_during_accumulation"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_errors_in_blobs_after_accumulation"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_errors_in_blobs_assigned_after_accumulation"] = {
        i: 0 for i in range(1, number_of_animals + 1)
    }
    results["number_of_individual_blobs"] = 0
    results["number_of_crossing_blobs"] = 0
    results["number_of_crossings_blobs_assigned_as_individuals"] = 0
    results["frames_with_identity_errors"] = []
    results["fragment_identifiers_with_identity_errors"] = []
    results["frames_with_crossing_errors"] = []
    results["fragment_identifiers_with_crossing_errors"] = []
    results["frames_with_zeros_in_groundtruth"] = []
    results["number_of_crossing_fragments"] = 0
    results["fragments_identifiers_of_crossings"] = []

    for gt_blobs_in_frame, blobs_in_frame in zip(
        blobs_in_video_groundtruth, blobs_in_video
    ):
        for blob in blobs_in_frame:
            corresponding_gt_blobs = get_corresponding_gt_blob(
                blob, gt_blobs_in_frame
            )
            if blob.is_an_individual and len(corresponding_gt_blobs) == 1:
                groundtruth_blob = corresponding_gt_blobs[0]
                if (
                    groundtruth_blob.is_an_individual
                    and groundtruth_blob.identity != -1
                    and not groundtruth_blob.was_a_crossing
                ):  # we are not considering crossing or failures of the model area
                    if identities_dictionary_permutation is not None:
                        gt_identity = identities_dictionary_permutation[
                            groundtruth_blob.identity
                        ]
                    else:
                        gt_identity = groundtruth_blob.identity
                    results["number_of_individual_blobs"] += 1
                    if gt_identity == 0:
                        results["frames_with_zeros_in_groundtruth"].append(
                            groundtruth_blob.frame_number
                        )
                    else:
                        try:
                            if (
                                blob.assigned_identity != 0
                                and blob.identity_corrected_closing_gaps
                                is None
                            ):  # we only consider P2 for non interpolated blobs
                                results["sum_individual_P2"][
                                    gt_identity
                                ] += blob._P2_vector[gt_identity - 1]
                        except:
                            logger.debug("P2_vector %s" % str(blob._P2_vector))
                            logger.debug(
                                "individual %s" % str(blob.is_an_individual)
                            )
                            logger.debug(
                                "fragment identifier ",
                                blob.fragment_identifier,
                            )
                        results["number_of_blobs_per_identity"][
                            gt_identity
                        ] += 1
                        results["number_of_assigned_blobs_per_identity"][
                            gt_identity
                        ] += (1 if blob.assigned_identity != 0 else 0)
                        results[
                            "number_of_blobs_assigned_during_accumulation_per_identity"
                        ][gt_identity] += (1 if blob.used_for_training else 0)
                        results[
                            "number_of_blobs_after_accumulation_per_identity"
                        ][gt_identity] += (
                            1 if not blob.used_for_training else 0
                        )
                        if gt_identity != blob.assigned_identity:
                            results["number_of_errors_in_all_blobs"][
                                gt_identity
                            ] += 1
                            results[
                                "number_of_errors_in_blobs_after_accumulation"
                            ][gt_identity] += (
                                1 if not blob.used_for_training else 0
                            )
                            if blob.assigned_identity != 0:
                                results["number_of_errors_in_assigned_blobs"][
                                    gt_identity
                                ] += 1
                                results[
                                    "number_of_errors_in_blobs_assigned_during_accumulation"
                                ][gt_identity] += (
                                    1 if blob.used_for_training else 0
                                )
                                results[
                                    "number_of_errors_in_blobs_assigned_after_accumulation"
                                ][gt_identity] += (
                                    1 if not blob.used_for_training else 0
                                )
                            if (
                                blob.fragment_identifier
                                not in results[
                                    "fragment_identifiers_with_identity_errors"
                                ]
                            ):
                                results["frames_with_identity_errors"].append(
                                    blob.frame_number
                                )
                                results[
                                    "fragment_identifiers_with_identity_errors"
                                ].append(blob.fragment_identifier)

                elif groundtruth_blob.is_a_crossing or gt_identity == -1:
                    if (
                        blob.fragment_identifier
                        not in results["fragments_identifiers_of_crossings"]
                    ):
                        results["fragments_identifiers_of_crossings"].append(
                            blob.fragment_identifier
                        )
                        results["number_of_crossing_fragments"] += 1
                    results["number_of_crossing_blobs"] += 1
                    results[
                        "number_of_crossings_blobs_assigned_as_individuals"
                    ] += (1 if blob.is_an_individual else 0)
                    if blob.is_an_individual:
                        if (
                            blob.fragment_identifier
                            not in results[
                                "fragment_identifiers_with_crossing_errors"
                            ]
                        ):
                            results["frames_with_crossing_errors"].append(
                                blob.frame_number
                            )
                            results[
                                "fragment_identifiers_with_crossing_errors"
                            ].append(blob.fragment_identifier)

    return results


def check_ground_truth_consistency(video, gt_video):
    if video.number_of_frames != gt_video.number_of_frames:
        raise ValueError(
            "The number of frames in the video and in the \
                         groundtruth video are different. The groundtruth\
                         file cannot be reused"
        )


def check_first_frame_first_global_fragment(
    video,
    first_frame_first_global_fragment,
    blobs_in_video_groundtruth,
    blobs_in_video,
):
    while (
        len(blobs_in_video_groundtruth[first_frame_first_global_fragment])
        < video.number_of_animals
    ):
        first_frame_first_global_fragment += 1
    return first_frame_first_global_fragment


# def get_permutation_of_identities(video,
#                                   first_frame_first_global_fragment,
#                                   blobs_in_video_groundtruth,
#                                   blobs_in_video):
#     print(first_frame_first_global_fragment)
#     first_frame_first_global_fragment = check_first_frame_first_global_fragment(video, first_frame_first_global_fragment, blobs_in_video_groundtruth, blobs_in_video)
#     print(first_frame_first_global_fragment)
#     print(len(blobs_in_video_groundtruth[first_frame_first_global_fragment]))
#     if first_frame_first_global_fragment is not None:
#         groundtruth_identities_in_first_frame = \
#             [blob.identity for blob in
#              blobs_in_video_groundtruth[first_frame_first_global_fragment]]
#         identities_in_first_frame = \
#             [blob.identity for blob in
#              blobs_in_video[first_frame_first_global_fragment]]
#         logger.debug('groundtruth identities in first frame %s'
#                      % str(groundtruth_identities_in_first_frame))
#         logger.debug('identities in first frame %s'
#                      % str(identities_in_first_frame))
#         identities_dictionary_permutation = \
#             {groundtruth_identity: identity for identity, groundtruth_identity
#              in zip(identities_in_first_frame,
#                     groundtruth_identities_in_first_frame)}
#     else:
#         identities_dictionary_permutation = None
#     print("identities_dictionary_permutation:",
#           identities_dictionary_permutation)
#
#     return identities_dictionary_permutation


def get_permutation_of_identities(
    video,
    first_frame_first_global_fragment,
    blobs_in_video_groundtruth,
    blobs_in_video,
):
    first_frame_first_global_fragment = check_first_frame_first_global_fragment(
        video,
        first_frame_first_global_fragment,
        blobs_in_video_groundtruth,
        blobs_in_video,
    )

    matching_found = False
    while not matching_found:
        gt_blobs_in_first_frame = blobs_in_video_groundtruth[
            first_frame_first_global_fragment
        ]
        identities_dictionary_permutation = {}
        print(first_frame_first_global_fragment)
        print(len(blobs_in_video[first_frame_first_global_fragment]))
        print(
            len(blobs_in_video_groundtruth[first_frame_first_global_fragment])
        )
        for blob in blobs_in_video[first_frame_first_global_fragment]:
            corresponding_blobs = get_corresponding_gt_blob(
                blob, gt_blobs_in_first_frame
            )
            if len(corresponding_blobs) == 1:
                identities_dictionary_permutation[
                    corresponding_blobs[0].identity
                ] = blob.identity
            else:
                first_frame_first_global_fragment += 1
                break
        if len(identities_dictionary_permutation) == video.number_of_animals:
            matching_found = True
    print(identities_dictionary_permutation)
    return identities_dictionary_permutation


def get_accuracy_wrt_groundtruth(
    video,
    gt_video,
    blobs_in_video_groundtruth,
    blobs_in_video=None,
    identities_dictionary_permutation=None,
):
    number_of_animals = video.number_of_animals
    blobs_in_video = (
        blobs_in_video_groundtruth
        if blobs_in_video is None
        else blobs_in_video
    )
    results = compare_tracking_against_groundtruth(
        number_of_animals,
        blobs_in_video_groundtruth,
        blobs_in_video,
        identities_dictionary_permutation,
    )
    pprint(results)
    if len(results["frames_with_zeros_in_groundtruth"]) == 0:
        accuracies = {}
        accuracies["percentage_of_unoccluded_images"] = results[
            "number_of_individual_blobs"
        ] / (
            results["number_of_individual_blobs"]
            + results["number_of_crossing_blobs"]
        )
        accuracies["individual_P2_in_validated_part"] = {
            i: results["sum_individual_P2"][i]
            / results["number_of_blobs_per_identity"][i]
            for i in range(1, number_of_animals + 1)
        }
        accuracies["mean_individual_P2_in_validated_part"] = np.sum(
            results["sum_individual_P2"].values()
        ) / np.sum(results["number_of_blobs_per_identity"].values())
        accuracies["individual_accuracy"] = {
            i: 1
            - results["number_of_errors_in_all_blobs"][i]
            / results["number_of_blobs_per_identity"][i]
            if results["number_of_blobs_per_identity"] != 0
            else None
            for i in range(1, number_of_animals + 1)
        }
        accuracies["accuracy"] = 1.0 - np.sum(
            results["number_of_errors_in_all_blobs"].values()
        ) / np.sum(results["number_of_blobs_per_identity"].values())
        accuracies["individual_accuracy_assigned"] = {
            i: 1
            - results["number_of_errors_in_assigned_blobs"][i]
            / results["number_of_assigned_blobs_per_identity"][i]
            if results["number_of_assigned_blobs_per_identity"] != 0
            else None
            for i in range(1, number_of_animals + 1)
        }
        accuracies["accuracy_assigned"] = 1.0 - np.sum(
            results["number_of_errors_in_assigned_blobs"].values()
        ) / np.sum(results["number_of_assigned_blobs_per_identity"].values())
        # accuracies['individual_accuracy_in_accumulation'] = {i:1 - results['number_of_errors_in_blobs_assigned_during_accumulation'][i]/results['number_of_blobs_assigned_during_accumulation_per_identity'][i]
        #                         if results['number_of_blobs_assigned_during_accumulation_per_identity'] != 0 else None
        #                         for i in range(1, number_of_animals + 1)}
        accuracies["individual_accuracy_in_accumulation"] = {}
        for i in range(1, number_of_animals + 1):
            if (
                results[
                    "number_of_blobs_assigned_during_accumulation_per_identity"
                ][i]
                != 0
            ):
                accuracies["individual_accuracy_in_accumulation"][i] = (
                    1
                    - results[
                        "number_of_errors_in_blobs_assigned_during_accumulation"
                    ][i]
                    / results[
                        "number_of_blobs_assigned_during_accumulation_per_identity"
                    ][i]
                )
            else:
                accuracies["individual_accuracy_in_accumulation"][i] = None
        accuracies["accuracy_in_accumulation"] = 1.0 - np.sum(
            results[
                "number_of_errors_in_blobs_assigned_during_accumulation"
            ].values()
        ) / np.sum(
            results[
                "number_of_blobs_assigned_during_accumulation_per_identity"
            ].values()
        )
        accuracies["individual_accuracy_after_accumulation"] = {}
        for i in range(1, number_of_animals + 1):
            if (
                results["number_of_blobs_after_accumulation_per_identity"][i]
                != 0
            ):
                accuracies["individual_accuracy_after_accumulation"][i] = (
                    1
                    - results["number_of_errors_in_blobs_after_accumulation"][
                        i
                    ]
                    / results[
                        "number_of_blobs_after_accumulation_per_identity"
                    ][i]
                )
            else:
                accuracies["individual_accuracy_after_accumulation"][i] = None
        if (
            np.sum(
                results[
                    "number_of_blobs_after_accumulation_per_identity"
                ].values()
            )
            != 0
        ):
            accuracies["accuracy_after_accumulation"] = 1.0 - np.sum(
                results[
                    "number_of_errors_in_blobs_after_accumulation"
                ].values()
            ) / np.sum(
                results[
                    "number_of_blobs_after_accumulation_per_identity"
                ].values()
            )
        else:
            accuracies["accuracy_after_accumulation"] = None
        if results["number_of_crossing_blobs"] != 0:
            accuracies["crossing_detector_accuracy"] = (
                1.0
                - results["number_of_crossings_blobs_assigned_as_individuals"]
                / results["number_of_crossing_blobs"]
            )
        else:
            accuracies["crossing_detector_accuracy"] = None
        logger.info("accuracies %s" % str(accuracies))
        logger.info(
            "number of crossing fragments in ground truth interval: %i"
            % results["number_of_crossing_fragments"]
        )
        logger.info(
            "number of crossing blobs in ground truth interval: %i"
            % results["number_of_crossing_blobs"]
        )
        return accuracies, results

    else:
        logger.info(
            "there are fish with 0 identity in frame %s"
            % str(results["frames_with_zeros_in_groundtruth"])
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


def reduce_resolution_groundtruth_blobs(
    groundtruth, video, blobs_in_video_groundtruth
):
    gt_width, gt_height = groundtruth.video.width, groundtruth.video.height
    resolution_reduction = video.resolution_reduction
    for blobs_in_frame in blobs_in_video_groundtruth:
        for blob in blobs_in_frame:
            blob.pixels = reduce_pixels(
                blob,
                video.original_width,
                video.original_height,
                video.width,
                video.height,
                video.resolution_reduction,
            )


def compute_and_save_session_accuracy_wrt_groundtruth(
    video, groundtruth_type=None
):
    logger.info("loading list_of_blobs")
    if groundtruth_type == "normal":
        list_of_blobs = ListOfBlobs.load(video, video.blobs_path)
    elif groundtruth_type == "no_gaps":
        list_of_blobs = ListOfBlobs.load(video, video.blobs_no_gaps_path)
    # select ground truth file
    logger.info("loading groundtruth")
    if groundtruth_type == "normal":
        groundtruth_path = os.path.join(video.video_folder, "_groundtruth.npy")
    elif groundtruth_type == "no_gaps":
        groundtruth_path = os.path.join(
            video.video_folder, "_groundtruth_with_crossing_identified.npy"
        )
    groundtruth = np.load(groundtruth_path, allow_pickle=True).item()

    check_ground_truth_consistency(video, groundtruth.video)
    accumulation_number = int(video.accumulation_folder[-1])
    if video.resolution_reduction != 1:
        reduce_resolution_groundtruth_blobs(
            groundtruth, video, groundtruth.blobs_in_video
        )
    identities_dictionary_permutation = get_permutation_of_identities(
        video,
        video.first_frame_first_global_fragment[accumulation_number],
        groundtruth.blobs_in_video,
        list_of_blobs.blobs_in_video,
    )
    blobs_in_video_groundtruth = groundtruth.blobs_in_video[
        groundtruth.start : groundtruth.end
    ]
    blobs_in_video = list_of_blobs.blobs_in_video[
        groundtruth.start : groundtruth.end
    ]
    logger.info("computing groundtruth")
    if groundtruth_type == "normal":
        accuracies, results = get_accuracy_wrt_groundtruth(
            video,
            groundtruth.video,
            blobs_in_video_groundtruth,
            blobs_in_video,
            identities_dictionary_permutation,
        )
    elif groundtruth_type == "no_gaps":
        accuracies, results = get_accuracy_wrt_groundtruth_no_gaps(
            video,
            groundtruth,
            blobs_in_video_groundtruth,
            blobs_in_video,
            identities_dictionary_permutation,
        )
    if accuracies is not None:
        logger.info("saving accuracies in video")
        video.gt_start_end = (groundtruth.start, groundtruth.end)
        if groundtruth_type == "normal":
            video.gt_accuracy = accuracies
            video.gt_results = results
        elif groundtruth_type == "no_gaps":
            video.gt_accuracy_no_gaps = accuracies
            video.gt_results_no_gaps = results
        video.save()
    return video, groundtruth


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gt",
        "--groundtruth_type",
        type=str,
        help="type of groundtruth to compute \
                        ('no_gaps' or 'normal')",
    )
    parser.add_argument(
        "-sf", "--session_folder", type=str, help="path to the session folder"
    )
    args = parser.parse_args()

    groundtruth_type = args.groundtruth_type
    session_folder = args.session_folder
    video_object_path = os.path.join(session_folder, "video_object.npy")
    logger.info("loading video object")
    video = np.load(video_object_path, allow_pickle=True).item(0)
    video.update_paths(video_object_path)
    groundtruth_path = os.path.join(video.video_folder, "_groundtruth.npy")
    groundtruth = np.load(groundtruth_path, allow_pickle=True).item()
    video, groundtruth = compute_and_save_session_accuracy_wrt_groundtruth(
        video, groundtruth_type
    )
    print(video.gt_accuracy)
