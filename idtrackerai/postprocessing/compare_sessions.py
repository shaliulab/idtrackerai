import os
import numpy as np
import logging

# create logger with 'spam_application'
logger = logging.getLogger('compare_sessions')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('compare_sessions.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
cformatter = logging.Formatter('%(levelname)s - %(message)s')
fh.setFormatter(fformatter)
ch.setFormatter(cformatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def video_object(session):
    return os.path.join(session, 'video_object.npy')


def lists_of_fragments(session):
    return os.path.join(session, 'preprocessing', 'fragments.npy')


def lists_of_global_fragments(session):
    return os.path.join(session, 'preprocessing', 'global_fragments.npy')


def compare_folders_structure(sessions):
    def log_session_subdirectories(dir):
        logger.debug("----Session: {}".format(dir[0]))
        logger.debug("Folders in session: {}".format(len(dir[1])))
        logger.debug(dir[1])
        logger.debug("Files in session: {}".format(len(dir[2])))
        logger.debug(dir[2])

    logger.info("*****Comparing NUMBER OF SUBDIRECTORIES IN SESSIONS")
    number_of_folders = []
    dirs = []
    for session in sessions:
        dir = list(os.walk(session))[0]
        dirs.append(dir)
        number_of_folders.append(len(dir[1]))

    if len(set(number_of_folders)) > 1:
        logger.warning("The number of subdirectories in not the same in all sessions")
        logger.warning("number_of_folders: {}".format(number_of_folders))
        [log_session_subdirectories(dir) for dir in dirs]
    else:
        logger.info("All sessions have the same number of folders")


def compare_tracking_parameters(video_objects):
    logger.info("*****Comparing TRACKING PARAMETERS")
    list_of_params = ['resolution_reduction', 'apply_ROI', 'subtract_bkg',
                      'min_threshold', 'max_threshold',
                      'min_area', 'max_area',
                      'number_of_animals',
                      'tracking_interval']
    params_dict = {}
    for param in list_of_params:
        if param is 'tracking_interval':
            params_dict['has_tracking_interval'] = [getattr(v, param) is not None for v in video_objects] ###TODO: Get start end of tracking intervals
        else:
            params_dict[param] = [getattr(v, param) for v in video_objects]
        if param is 'subtract_bkg' and any(params_dict[param]):
            params_dict['bkg'] = [v.bkg for v in video_objects]

    for param in params_dict:
        if param is not 'bkg' and len(set(params_dict[param])) > 1:
            logger.warning("The parameter {} is not the same in all sessions".format(param))
            if param is not 'has_tracking_interval':
                logger.warning(params_dict[param])
            else:
                logger.warning("tracking intervals: {}".format([v.tracking_interval for v in video_objects]))
        elif param is 'bkg':
            logger.warning("Comparison of background to be implemented, they could be different")
        else:
            logger.info("The parameter {} is the same in all sessions".format(param))


def compare_tracked_video(video_objects):
    logger.info("*****Comparing VIDEOS TRACKED")
    videos_tracked = [os.path.split(v.video_path)[1] for v in video_objects]
    if len(set(videos_tracked)) > 1:
        logger.warning("The video tracked in not the same in all sessions")
        logger.warning(videos_tracked)
    else:
        logger.info("The video tracked is the same in all sessions")


def compare_finished_events(video_objects):
    logger.info("*****Comparing FINISHED EVENTS")
    list_of_params = ['has_preprocessing_parameters',
                      'has_been_segmented', 'has_been_preprocessed',
                      'first_accumulation_finished', 'has_been_pretrained',
                      'second_accumulation_finished',
                      'has_been_assigned',
                      'has_duplications_solved',
                      'has_trajectories',
                      'has_crossings_solved',
                      'has_trajectories_wo_gaps']
    params_dict = {}
    for param in list_of_params:
        params_dict[param] = [getattr(v, param) for v in video_objects]

    for param in params_dict:
        if len(set(params_dict[param])) > 1:
            logger.warning("The parameter {} is not the same in all sessions".format(param))
            logger.warning(params_dict[param])
        else:
            logger.info("The parameter {} is the same in all sessions".format(param))


def compare_preprocessing_variables(video_objects):
    logger.info("*****Comparing PREPROCESSING VARIABLES")
    list_of_params = ['maximum_number_of_blobs',
                      'median_body_length',
                      'model_area',
                      'number_of_episodes',
                      'there_are_crossings',
                      'crossing_image_size',
                      'identification_image_size',
                      'first_frame_first_global_fragment',
                      'maximum_number_of_images_in_global_fragments',
                      'individual_fragments_stats',
                      'number_of_global_fragments_candidates_for_accumulation',
                      'number_of_global_fragments',
                      'number_of_unique_images_in_global_fragments']


    params_dict = {}
    for param in list_of_params:
        if param is 'model_area':
            params_dict['median_area'] = [v.model_area.median for v in video_objects]
            params_dict['std_area'] = [v.model_area.std for v in video_objects]
        elif param is 'first_frame_first_global_fragment':
            params_dict[param] = [getattr(v,param)[0] for v in video_objects]
        elif param is 'individual_fragments_stats':
            for indiv_fragments_stat in video_objects[0].individual_fragments_stats:
                params_dict[indiv_fragments_stat] = [v.individual_fragments_stats[indiv_fragments_stat] for v in video_objects]
        else:
            params_dict[param] = [getattr(v, param) for v in video_objects]

    for param in params_dict:
        if len(set(params_dict[param])) > 1:
            logger.warning("The parameter {} is not the same in all sessions".format(param))
            logger.warning(params_dict[param])
        else:
            logger.info("The parameter {} is the same in all sessions".format(param))


def compare_tracking_variables(video_objects):
    logger.info("*****Comparing TRACKING VARIABLES")
    list_of_params = ['track_wo_identities',
                      'accumulation_trial',
                      'accumulation_step',
                      'accumulation_statistics',
                      'percentage_of_accumulated_images',
                      'ratio_accumulated_images',
                      'ratio_of_accumulated_images',
                      'overall_P2']
    list_of_accumulation_statistics_attributes= ['number_of_accumulated_global_fragments',
                                                 'number_of_non_certain_global_fragments',
                                                 'number_of_randomly_assigned_global_fragments',
                                                 'number_of_nonconsistent_global_fragments',
                                                 'number_of_nonunique_global_fragments',
                                                 'number_of_acceptable_global_fragments']


    params_dict = {}
    for param in list_of_params:
        if param is 'accumulation_statistics':
            for i, acumulation_stat in enumerate(list_of_accumulation_statistics_attributes):
                params_dict[acumulation_stat] = [tuple(v.accumulation_statistics[v.accumulation_trial][i]) for v in video_objects]
        elif param is 'percentage_of_accumulated_images':
            params_dict[param] = [getattr(v, param)[v.accumulation_trial] for v in video_objects]
        elif param is 'ratio_of_accumulated_images':
            params_dict[param] = [tuple(getattr(v, param)) for v in video_objects]
        else:
            params_dict[param] = [getattr(v, param) for v in video_objects]
    for param in params_dict:
        if len(set(params_dict[param])) > 1:
            logger.warning("The parameter {} is not the same in all sessions".format(param))
            logger.warning(params_dict[param])
        else:
            logger.info("The parameter {} is the same in all sessions".format(param))


def compare_list_of_fragments(list_of_fragmentss, list_of_global_fragmentss):
    logger.info("*****Comparing LIST OF FRAGMENTS")
    list_of_fragments_stats = [lf.get_stats(gf) for (lf, gf) in zip(list_of_fragmentss, list_of_global_fragmentss)]

    params_dict = {param: [stats[param] for stats in list_of_fragments_stats] for param in list_of_fragments_stats[0]}
    for param in params_dict:
        if len(set(params_dict[param])) > 1:
            logger.warning("The parameter {} is not the same in all sessions".format(param))
            logger.warning(params_dict[param])
        else:
            logger.info("The parameter {} is the same in all sessions".format(param))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sessions", nargs='+', type=str)
    args = parser.parse_args()
    logger.info("Comparing sessions: {}\n".format(args.sessions))
    video_objects = [np.load(video_object(s), allow_pickle=True).item() for s in args.sessions]
    compare_tracked_video(video_objects)
    compare_tracking_parameters(video_objects)
    compare_folders_structure(args.sessions)
    compare_finished_events(video_objects)
    compare_preprocessing_variables(video_objects)
    compare_tracking_variables(video_objects)
    list_of_fragmentss = [np.load(lists_of_fragments(s), allow_pickle=True).item() for s in args.sessions]
    list_of_global_fragmentss = [np.load(lists_of_global_fragments(s), allow_pickle=True).item() for s in args.sessions]
    compare_list_of_fragments(list_of_fragmentss, list_of_global_fragmentss)
