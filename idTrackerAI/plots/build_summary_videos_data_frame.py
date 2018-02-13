from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
sys.path.append('./groundtruth_utils')
sys.path.append('./postprocessing')
sys.path.append('./plots')
sys.path.append('./network/identification_model')

import pandas
import numpy as np
from pprint import pprint
import pandas as pd
from glob import glob

from video import Video
from list_of_blobs import ListOfBlobs
from list_of_fragments import ListOfFragments
from list_of_global_fragments import ListOfGlobalFragments
from generate_groundtruth import GroundTruthBlob, GroundTruth
from generate_individual_groundtruth import GroundTruthBlob, IndividualGroundTruth, generate_individual_groundtruth
from compute_groundtruth_statistics import get_accuracy_wrt_groundtruth
from compute_individual_groundtruth_statistics import get_individual_accuracy_wrt_groundtruth
from identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities
from global_fragments_statistics import compute_and_plot_fragments_statistics

sessions = ['100 drosophila (females)/Canton_N100_11-23-17_12-59-17/session_20180122',
    '10_fish_group4/first/session_20180122',
    '10_fish_group5/first/session_20180131',
    '10_fish_group6/first/session_20180202',
    '10_flies_compressed_clara/session_20180207',
    '38 drosophila (females males)/Canton_N38_top_video_01-31-18_10-50-14/session_20180201',
    '60 drosophila (females)/Canton_N59_12-15-17_16-32-02/session_20180102',
    '60 drosophila (females)/Canton_N60_12-15-17_15-15-10/session_20171221',
    '72 drosophila (females - males)/session_20180201',
    '80 drosophila (females - males)/session_20180206',
    '80 drosophila (females males)/Canton_N80_11-28-17_17-21-32/session_20180123',
    'ants_andrew_1/session_20180206',
    'idTrackerDeep_LargeGroups_1/100/First/session_20180102',
    'idTrackerDeep_LargeGroups_1/60/First/session_20180108',
    'idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_100/First/session_20180104',
    'idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_60/First/session_20171221',
    'idTrackerDeep_LargeGroups_3/100fish/First/session_02122017',
    'idTrackerDeep_LargeGroups_3/60fish/First/session_20171225',
    'idTrackerVideos/8zebrafish_conflicto/session_20180130',
    'idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo10/session_20180201',
    'idTrackerVideos/Hipertec_pesados/Medaka/2012may31/Grupo5/session_20180131',
    'idTrackerVideos/Hipertec_pesados/Medaka/20fish_contapa/session_20180201',
    'idTrackerVideos/Moscas/2011dic12/Video_4fem_2mal_bottom/session_20180130',
    'idTrackerVideos/Moscas/20121010/PlatoGrande_8females_2/session_20180131',
    'idTrackerVideos/NatureMethods/Isogenicos/Wik_8_grupo4/session_20180130',
    'idTrackerVideos/NatureMethods/Ratones4/session_20180205',
    'idTrackerVideos/NatureMethods/VideoRatonesDespeinaos3/session_20180206',
    'idTrackerVideos/Ratones/20121203/2aguties/session_20180204',
    'idTrackerVideos/Ratones/20121203/2negroscanosos/session_20180204',
    'idTrackerVideos/Ratones/20121203/2negroslisocanoso/session_20180205',
    'idTrackerVideos/Ratones/20121203/2negroslisos/session_20180205',
    'idTrackerVideos/ValidacionTracking/Moscas/Platogrande_8females/session_20180131',
    'idTrackerVideos/Zebrafish_nacreLucie/pair3ht/session_20180207']

animal_type = ['drosophila (females)', 'zebrafish (30dpf)', 'zebrafish (30dpf)', 'zebrafish (30dpf)','drosophila',
            'drosophila', 'drosophila (females)', 'drosophila (females)', 'drosophila', 'drosophila',
            'drosophila', 'ants', 'zebrafish (30dpf)', 'zebrafish (30dpf)', 'zebrafish (30dpf)',
            'zebrafish (30dpf)', 'zebrafish (30dpf)', 'zebrafish (30dpf)', 'zebrafish', 'medaka',
            'medaka', 'medaka', 'drosophila', 'drosophila', 'zebrafish',
            'black mice', 'black mice', 'agouti mice', 'black mice', 'black mice',
            'black mice', 'drosophila', 'nacre zebrafish']

idTracker_video = [False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False, True, True,
                    True, True, True, True, True,
                    True, True, True, True, True,
                    True, True, True]

used_for_developing = [False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, True, False, False,
                    False, False, False, True, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False]

bad_video_example = [True, False, False, False, False,
                    False, False, True, False, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False, False, False,
                    False, False, False]


def get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video):
    if hasattr(video, 'accumulation_folder'):
        list_of_global_fragments.order_by_distance_travelled()
        global_fragment_for_accumulation = int(video.accumulation_folder[-1])
        if global_fragment_for_accumulation > 0:
            global_fragment_for_accumulation -= 1

        number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
        return np.min(number_of_images_in_fragments)
    else:
        return None

def get_mean_number_of_images_in_first_global_fragment(list_of_global_fragments, video):
    if hasattr(video, 'accumulation_folder'):
        list_of_global_fragments.order_by_distance_travelled()
        global_fragment_for_accumulation = int(video.accumulation_folder[-1])
        if global_fragment_for_accumulation > 0:
            global_fragment_for_accumulation -= 1

        number_of_images_in_fragments = list_of_global_fragments.global_fragments[global_fragment_for_accumulation].number_of_images_per_individual_fragment
        return np.mean(number_of_images_in_fragments)
    else:
        return None

if __name__ == '__main__':
    # hard_drive_path = '/media/themis/ground_truth_results_backup'
    path_to_results_hard_drive = '/media/prometheus/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    session_paths = [x[0] for x in os.walk(tracked_videos_folder) if 'session' in x[0][-16:] and 'Trash' not in x[0]]
    pprint(session_paths)
    tracked_videos_data_frame = pd.DataFrame()
    if len(session_paths) == len(sessions) and len(session_paths) == len(animal_type) and len(idTracker_video) == len(session_paths):

        for session_path in session_paths:
            print("\n******************************")
            print('Session: ', session_path)
            video_path = os.path.join(session_path,'video_object.npy')
            video_folder = os.path.split(session_path)[0]
            video = np.load(video_path).item(0)

            session_number = [index for index in range(len(sessions)) if sessions[index] in session_path]
            assert len(session_number) == 1
            session_number = session_number[0]
            ### give animal type name
            video.animal_type = animal_type[session_number]
            bad_video = bad_video_example[session_number]

            ### give if idTracker video or not
            video.idTracker_video = idTracker_video[session_number]

            ### create blobs_collection_interpolated if does not exist
            blobs_interpolated_path = os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy')
            if not os.path.isfile(blobs_interpolated_path):
                print("\ncreating list_of_blobs_interpolated")
                print("loading list_of_blobs")
                list_of_blobs = ListOfBlobs.load(video, os.path.join(session_path,'preprocessing', 'blobs_collection.npy'))
                print("loading list_of_blobs_no_gaps")
                list_of_blobs_no_gaps = ListOfBlobs.load(video, os.path.join(session_path,'preprocessing', 'blobs_collection_no_gaps.npy'))

                video._blobs_path_interpolated = os.path.join(video.preprocessing_folder, 'blobs_collection_interpolated.npy')
                list_of_blobs_interpolated = assign_zeros_with_interpolation_identities(list_of_blobs, list_of_blobs_no_gaps)
                list_of_blobs_interpolated.save(video, os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy'), number_of_chunks = video.number_of_frames)

            if not bad_video:
                if not hasattr(video, 'gt_accuracy_interpolated') or not hasattr(video, 'gt_results_interpolated'):
                    print("\ncomputing gt_accuracy_interpolated")
                    if not 'list_of_blobs_interpolated' in locals():
                        print("loading list_of_fragments")
                        list_of_blobs_interpolated = ListOfBlobs.load(video, os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy'))
                    print("loading ground truth file")
                    groundtruth = np.load(os.path.join(video_folder, '_groundtruth.npy')).item()
                    blobs_in_video_groundtruth = groundtruth.blobs_in_video[groundtruth.start:groundtruth.end]
                    blobs_in_video_interpolated = list_of_blobs_interpolated.blobs_in_video[groundtruth.start:groundtruth.end]
                    print("computing groundtruth")
                    accuracies, results = get_accuracy_wrt_groundtruth(video, blobs_in_video_groundtruth, blobs_in_video_interpolated)
                    print("saving video")
                    video.gt_accuracy_interpolated = accuracies
                    video.gt_results_interpolated = results

            if not hasattr(video, 'protocol'):
                print("\ncomputing protocol")
                if not video.has_been_pretrained and len(video.validation_accuracy) == 1:
                    video.protocol = 1
                elif not video.has_been_pretrained and len(video.validation_accuracy) >= 2:
                    video.protocol = 2
                elif video.has_been_pretrained:
                    video.protocol = 3

            if not hasattr(video, '_gamma_fit_parameters'):
                print("\ncomputing gamma_fit_parameters")
                if not 'list_of_blobs' in locals():
                    print("loading list_of_blobs")
                    list_of_blobs = ListOfBlobs.load(video, os.path.join(session_path,'preprocessing', 'blobs_collection.npy'))
                if not 'list_of_fragments' in locals():
                    print("loading list_of_fragments")
                    list_of_fragments = ListOfFragments.load(os.path.join(session_path, 'preprocessing', 'fragments.npy'))
                if not 'list_of_global_fragments' in locals():
                    print("loading list_of_global_fragments")
                    list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'), list_of_fragments.fragments)
                video.individual_fragments_lenghts, \
                video.individual_fragments_distance_travelled, \
                video._gamma_fit_parameters = compute_and_plot_fragments_statistics(video,
                                                                                    video.model_area,
                                                                                    list_of_blobs,
                                                                                    list_of_fragments,
                                                                                    list_of_global_fragments,
                                                                                    plot = False,
                                                                                    save = False)

            if not hasattr(video, 'individual_fragments_stats'):
                print("\ncomputing individual_fragments_stats")
                if not 'list_of_fragments' in locals():
                    print("loading list_of_fragments")
                    list_of_fragments = ListOfFragments.load(os.path.join(session_path, 'preprocessing', 'fragments.npy'))
                if not 'list_of_global_fragments' in locals():
                    print("loading list_of_global_fragments")
                    list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'), list_of_fragments.fragments)
                video.individual_fragments_stats = list_of_fragments.get_stats(list_of_global_fragments)

            if not hasattr(video, 'number_of_images_in_shortest_fragment_in_first_global_fragment') or \
                video.number_of_images_in_shortest_fragment_in_first_global_fragment is None:
                print("\ncomputing number_of_images_in_shortest_fragment_in_first_global_fragment")
                if not 'list_of_fragments' in locals():
                    print("loading list_of_fragments")
                    list_of_fragments = ListOfFragments.load(os.path.join(session_path, 'preprocessing', 'fragments.npy'))
                if not 'list_of_global_fragments' in locals():
                    print("loading list_of_global_fragments")
                    list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'), list_of_fragments.fragments)
                video.number_of_images_in_shortest_fragment_in_first_global_fragment = get_number_of_images_in_shortest_fragment_in_first_global_fragment(list_of_global_fragments, video)
                video.mean_number_of_images_in_first_global_fragment = get_mean_number_of_images_in_first_global_fragment(list_of_global_fragments, video)

                print(video.number_of_images_in_shortest_fragment_in_first_global_fragment)
                print(video.mean_number_of_images_in_first_global_fragment)

            if not hasattr(video, 'number_of_global_fragments'):
                if not 'list_of_fragments' in locals():
                    print("loading list_of_fragments")
                    list_of_fragments = ListOfFragments.load(os.path.join(session_path, 'preprocessing', 'fragments.npy'))
                if not 'list_of_global_fragments' in locals():
                    print("loading list_of_global_fragments")
                    list_of_global_fragments = ListOfGlobalFragments.load(os.path.join(session_path, 'preprocessing', 'global_fragments.npy'), list_of_fragments.fragments)
                video.number_of_global_fragments = list_of_global_fragments.number_of_global_fragments

            if not bad_video:
                individual_groundtruth_paths = glob(os.path.join(video_folder,'_individual*.npy'))
                if not hasattr(video, 'individual_groundtruths') and len(individual_groundtruth_paths) != 0:
                    if not 'list_of_blobs' in locals():
                        print("loading list_of_blobs")
                        list_of_blobs = ListOfBlobs.load(video, os.path.join(session_path, 'preprocessing', 'blobs_collection.npy'))

                    print("updating video with individual_groundtruths")
                    individual_groundtruth_paths = glob(os.path.join(video_folder,'_individual*.npy'))
                    video.individual_groundtruths = []
                    for individual_groundtruth_path in individual_groundtruth_paths:
                        groundtruth = np.load(individual_groundtruth_path).item()

                        individual_blobs_in_video_groundtruth = [blob for blob in groundtruth.individual_blobs_in_video
                                                        if (blob.frame_number >= groundtruth.start
                                                        and blob.frame_number <= groundtruth.end)]
                        comparison_info = get_individual_accuracy_wrt_groundtruth(video, individual_blobs_in_video_groundtruth)
                        comparison_info['number_of_occluded_frames'] = groundtruth.end - groundtruth.start - len(individual_blobs_in_video_groundtruth)
                        pprint(comparison_info)
                        comparison_info['start-end'] = (groundtruth.start, groundtruth.end)
                        video.individual_groundtruths.append(comparison_info)

                if hasattr(video, 'individual_groundtruths') and not hasattr(video, 'individual_groundtruths_interpolated'):
                    if not 'list_of_blobs_interpolated' in locals():
                        print("loading list_of_blobs_interpolated")
                        list_of_blobs_interpolated = ListOfBlobs.load(video, os.path.join(session_path, 'preprocessing', 'blobs_collection_interpolated.npy'))

                    print("computing individual_groundtruths_interpolated")
                    individual_groundtruth_paths = glob(os.path.join(video_folder,'_individual*.npy'))
                    video.individual_groundtruths_interpolated = []
                    for individual_groundtruth_path in individual_groundtruth_paths:
                        groundtruth = np.load(individual_groundtruth_path).item()

                        groundtruth = generate_individual_groundtruth (video, blobs_in_video = list_of_blobs_interpolated.blobs_in_video,
                                                            start = groundtruth.start, end = groundtruth.end,
                                                            validated_identity = groundtruth.validated_identity, save_gt = False)

                        individual_blobs_in_video_groundtruth = [blob for blob in groundtruth.individual_blobs_in_video
                                                        if (blob.frame_number >= groundtruth.start
                                                        and blob.frame_number <= groundtruth.end)]
                        comparison_info = get_individual_accuracy_wrt_groundtruth(video, individual_blobs_in_video_groundtruth)
                        comparison_info['number_of_occluded_frames'] = groundtruth.end - groundtruth.start - len(individual_blobs_in_video_groundtruth)
                        pprint(comparison_info)
                        comparison_info['start-end'] = (groundtruth.start, groundtruth.end)
                        video.individual_groundtruths_interpolated.append(comparison_info)

                if hasattr(video, 'individual_groundtruths') and hasattr(video, 'individual_groundtruths_interpolated'):
                    print("computing gt_accuracy_individual and gt_accuracy_individual_interpolated")
                    accuracies = []
                    accuracies_assigned = []
                    accuracies_interpolated = []
                    accuracies_assigned_interpolated = []

                    ### This way they do not necesarely refer to the same identity. I should check
                    for individual_gt, individual_gt_interpolated in zip(video.individual_groundtruths, video.individual_groundtruths_interpolated):
                        accuracies.append(individual_gt['accuracy'])
                        accuracies_assigned.append(individual_gt['accuracy_assigned'])
                        accuracies_interpolated.append(individual_gt_interpolated['accuracy'])
                        accuracies_assigned_interpolated.append(individual_gt_interpolated['accuracy_assigned'])
                    video.gt_accuracy_individual = {'accuracy': np.mean(accuracies),
                                                    'accuracy_assigned': np.mean(accuracies_assigned)}
                    video.gt_accuracy_individual_interpolated = {'accuracy': np.mean(accuracies_interpolated),
                                                                'accuracy_assigned': np.mean(accuracies_assigned_interpolated)}

            np.save(video_path, video)
            if 'list_of_blobs' in locals():
                del list_of_blobs
            if 'list_of_blobs_interpolated' in locals():
                del list_of_blobs_interpolated
            if 'list_of_fragments' in locals():
                del list_of_fragments
            if 'list_of_global_fragments' in locals():
                del list_of_global_fragments

            tracked_videos_data_frame = \
                tracked_videos_data_frame.append({'session_path': session_path,
                    'git_commit': video.git_commit,
                    'video_title': str(video.number_of_animals) + ' ' + video.animal_type,
                    'bad_video_example': bad_video,
                    'video_name': os.path.split(video.video_path)[1],
                    'animal_type': video.animal_type,
                    'idTracker_video': video.idTracker_video,
                    'used_for_developing': used_for_developing[session_number],
                    'number_of_animals': video.number_of_animals,
                    'number_of_frames': video.number_of_frames,
                    'frame_rate': video.frames_per_second,
                    'min_threshold': video.min_threshold,
                    'max_threshold': video.max_threshold,
                    'min_area': video.min_area,
                    'max_area': video.max_area,
                    'subtract_bkg': video.subtract_bkg,
                    'apply_ROI': video.apply_ROI,
                    'resolution_reduction': video.resolution_reduction,
                    'resegmentation_parameters': video.resegmentation_parameters,
                    'knowledge_transfer_model_folder': video.knowledge_transfer_model_folder,
                    'maximum_number_of_blobs': video.maximum_number_of_blobs,
                    'width': video.width,
                    'height': video.height,
                    'original_width': video.original_width,
                    'original_height': video.original_height,
                    'tracking_with_knowledge_transfer': video.tracking_with_knowledge_transfer,
                    'video_length_min': video.number_of_frames/video.frames_per_second/60,
                    'tracking_time': None if not hasattr(video, 'total_time') else video.total_time / 60,
                    'preprocessing_time': None if not hasattr(video, 'preprocessing_time') else video.total_time / 60,
                    'first_accumulation_time': None if not hasattr(video, 'first_accumulation_time') else video.total_time / 60,
                    'assignment_time': None if not hasattr(video, 'assignment_time') else video.total_time / 60,
                    'pretraining_time': None if not hasattr(video, 'pretraining_time') else video.total_time / 60,
                    'second_accumulation_time': None if not hasattr(video, 'second_accumulation_time') else video.total_time / 60,
                    'solve_impossible_jumps_time': None if not hasattr(video, 'solve_impossible_jumps_time') else video.total_time / 60,
                    'generate_trajectories_time': None if not hasattr(video, 'generate_trajectories_time') else video.total_time / 60,
                    'mean_area_in_pixels': video.model_area.mean,
                    'std_area_in_pixels': video.model_area.std,
                    'body_length': video.median_body_length,
                    'identification_image_size': video.identification_image_size,
                    'gamma_scale_parameter': video.gamma_fit_parameters[2],
                    'gamma_shape_parameter': video.gamma_fit_parameters[0],
                    'number_of_images_in_shortest_fragment_in_first_global_fragment': video.number_of_images_in_shortest_fragment_in_first_global_fragment,
                    'mean_number_of_images_in_first_global_fragment': video.mean_number_of_images_in_first_global_fragment,
                    'minimum_number_of_frames_moving_in_first_global_fragment': None if not hasattr(video, 'minimum_number_of_frames_moving_in_first_global_fragment') else video.minimum_number_of_frames_moving_in_first_global_fragment,
                    'number_of_global_fragments': video.number_of_global_fragments,
                    'number_of_fragments': video.individual_fragments_stats['number_of_fragments'],
                    'number_of_crossing_fragments': video.individual_fragments_stats['number_of_crossing_fragments'],
                    'number_of_individual_fragments': video.individual_fragments_stats['number_of_individual_fragments'],
                    'number_of_individual_fragments_not_in_a_global_fragment': video.individual_fragments_stats['number_of_individual_fragments_not_in_a_global_fragment'],
                    'number_of_not_accumulable_individual_fragments': video.individual_fragments_stats['number_of_not_accumulable_individual_fragments'],
                    'number_of_globally_accumulated_individual_blobs': video.individual_fragments_stats['number_of_globally_accumulated_individual_blobs'],
                    'number_of_partially_accumulated_individual_fragments': video.individual_fragments_stats['number_of_partially_accumulated_individual_fragments'],
                    'number_of_blobs': video.individual_fragments_stats['number_of_blobs'],
                    'number_of_crossing_blobs': video.individual_fragments_stats['number_of_crossing_blobs'],
                    'number_of_individual_blobs': video.individual_fragments_stats['number_of_individual_blobs'],
                    'number_of_individual_blobs_not_in_a_global_fragment': video.individual_fragments_stats['number_of_individual_blobs_not_in_a_global_fragment'],
                    'number_of_not_accumulable_individual_blobs': video.individual_fragments_stats['number_of_not_accumulable_individual_blobs'],
                    'number_of_globally_accumulated_individual_fragments': video.individual_fragments_stats['number_of_globally_accumulated_individual_fragments'],
                    'number_of_partially_accumulated_individual_blobs': video.individual_fragments_stats['number_of_partially_accumulated_individual_blobs'],
                    'protocol_used': video.protocol,
                    'accumulation_trial': video.accumulation_trial,
                    'number_of_accumulation_steps': len(video.validation_accuracy),
                    'percentage_of_accumulated_images': video.percentage_of_accumulated_images[video.accumulation_trial],
                    'estimated_accuracy': video.overall_P2,
                    'interval_of_frames_validated': video.gt_start_end if not bad_video else -1,
                    'number_of_frames_validated': np.diff(video.gt_start_end)[0] if not bad_video else -1,
                    'percentage_of_video_validated': np.diff(video.gt_start_end)[0]/video.number_of_frames*100 if not bad_video else -1,
                    'time_validated_min': np.diff(video.gt_start_end)[0]/video.frames_per_second/60 if not bad_video else -1,
                    'number_of_crossing_fragments_in_validated_part': video.gt_results['number_of_crossing_fragments'] if not bad_video else -1,
                    'number_of_crossing_images_in_validated_part': video.gt_results['number_of_crossing_blobs'] if not bad_video else -1,
                    'false_positive_rate_in_crossing_detector': 1. - video.gt_accuracy['crossing_detector_accuracy'] if not bad_video else -1,
                    'percentage_of_unoccluded_images': video.gt_accuracy['percentage_of_unoccluded_images'] if not bad_video else -1,
                    'estimated_accuracy_in_validated_part': video.gt_accuracy['mean_individual_P2_in_validated_part'] if not bad_video else -1,
                    'accuracy_in_accumulation': video.gt_accuracy['accuracy_in_accumulation'] if not bad_video else -1,
                    'accuracy_only_identification': video.gt_accuracy['accuracy'] if not bad_video else video.overall_P2,
                    'accuracy_identified_animals_only_identification': video.gt_accuracy['accuracy_assigned'] if not bad_video else -1,
                    'accuracy_in_residual_identification_only_identification': video.gt_accuracy['accuracy_after_accumulation'] if not bad_video else -1,
                    'accuracy_identification_and_interpolation':video.gt_accuracy_interpolated['accuracy'] if not bad_video else video.overall_P2,
                    'accuracy_identified_animals_identification_and_interpolation': video.gt_accuracy_interpolated['accuracy_assigned'] if not bad_video else -1,
                    'accuracy_in_residual_identification_identification_and_interpolation': video.gt_accuracy_interpolated['accuracy_after_accumulation'] if not bad_video else -1,
                    'accuracy_with_gaps_closed_by_interpolation': -1 if not hasattr(video, 'gt_accuracy_no_gaps') else video.gt_accuracy_no_gaps['accuracy'],
                    'individual_accuracy': -1 if not hasattr(video, 'gt_accuracy_individual') else video.gt_accuracy_individual['accuracy'],
                    'individual_accuracy_identified_animals': -1 if not hasattr(video, 'gt_accuracy_individual') else video.gt_accuracy_individual['accuracy_assigned'],
                    'individual_accuracy_interpolated': -1 if not hasattr(video, 'gt_accuracy_individual_interpolated') else video.gt_accuracy_individual_interpolated['accuracy'],
                    'individual_accurcay_identified_animals_interpolated': -1 if not hasattr(video, 'gt_accuracy_individual_interpolated') else video.gt_accuracy_individual_interpolated['accuracy_assigned']
                    }, ignore_index=True)

        tracked_videos_data_frame.to_pickle(os.path.join(tracked_videos_folder, 'tracked_videos_data_frame.pkl'))
    else:
        print("update the list of sessions and species")
