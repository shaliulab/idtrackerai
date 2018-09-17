import logging; logging.basicConfig(level=logging.INFO)

from idtrackerai.constants import THRESHOLD_ACCEPTABLE_ACCUMULATION
from idtrackerai.constants import THRESHOLD_EARLY_STOP_ACCUMULATION

from idtrackerai.video import Video, os
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.crossing_detector import detect_crossings
from idtrackerai.preprocessing.segmentation import segment
from idtrackerai.list_of_fragments import ListOfFragments, create_list_of_fragments
from idtrackerai.list_of_global_fragments import ListOfGlobalFragments, create_list_of_global_fragments
from idtrackerai.accumulator import perform_one_accumulation_step

from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.accumulation_manager import AccumulationManager

class ConsistanceException(Exception): pass

def step1_pre_processing(
    video, n_animals, chk_segm_consist, update_func=None):

    logging.info('create_preprocessing_folder()')

    video.create_preprocessing_folder()

    if update_func: update_func(1, total=7)

    # Find blobs
    logging.info('Segment the video to find blobs.')
    blobs = segment(video)

    list_of_blobs = ListOfBlobs(blobs_in_video=blobs)

    if update_func: update_func(2)

    # Check the number of blobs consistancy
    logging.info('Check blobs consistancy.')
    error_iframes, max_nblobs = list_of_blobs.check_maximal_number_of_blob( n_animals, return_maximum_number_of_blobs=True)

    if update_func: update_func(3)
    

    if len(error_iframes) > 0 and (chk_segm_consist or n_animals == 1):
        
        raise ConsistanceException(
            """On some frames it was detected more blobs than animals.
            Please adjust the segmentation parameters."""
        )

    else:
        logging.info('No consistancy errors were found.')

        video._has_been_segmented = True
        if len(list_of_blobs.blobs_in_video[-1])==0:
            list_of_blobs.blobs_in_video   = list_of_blobs.blobs_in_video[:-1]
            list_of_blobs.number_of_frames = len(list_of_blobs.blobs_in_video)
            video._number_of_frames        = list_of_blobs.number_of_frames
        
        # save data
        #video.save()
        logging.info('Save the list of blobs.')
        list_of_blobs.save(video, video.blobs_path_segmented, number_of_chunks=video.number_of_frames)
        
        logging.info('Create a area model of the blobs.')
        area, median_bodylen = list_of_blobs.compute_model_area_and_body_length(n_animals)

        if update_func: update_func(4)
        
        video._model_area = area 
        video._median_body_length = median_bodylen

        logging.info('Identify the image size.')
        video.compute_identification_image_size(median_bodylen)

        if update_func: update_func(5)
        
        logging.info('Check if the blobs are organized in fragments.')
        if not list_of_blobs.blobs_are_connected:
            logging.info('Blobs are not organized in fragments. Compute the fragments.')
            list_of_blobs.compute_overlapping_between_subsequent_frames()

        if update_func: update_func(6)

        if n_animals != 1:
            logging.info('Multiple animals: Detect crossing')
            crossing_detector_trainer = detect_crossings(
                list_of_blobs,
                video,
                video.model_area,
                use_network=True,
                return_store_objects=True,
                plot_flag=False
            )

        else:
            logging.info('A single animal: Detect crossing')
            list_of_blob = detect_crossings(
                list_of_blobs,
                video,
                video.model_area,
                use_network=False,
                return_store_objects=False,
                plot_flag=False
            )
            
            logging.info('Save the list of blobs.')
            list_of_blob.save(video, video.blobs_path_segmented, number_of_chunks=video.number_of_frames)
            crossing_detector_trainer = None

        if update_func: update_func(7)

        # TODO: Plot statistics

        if video.number_of_animals != 1:
            
            logging.info('Multiple animals: Detect crossing')
            
            list_of_blobs.compute_overlapping_between_subsequent_frames()
            n_animals = max(video.number_of_animals, video.maximum_number_of_blobs)
            list_of_blobs.compute_fragment_identifier_and_blob_index( n_animals )
            list_of_blobs.compute_crossing_fragment_identifier()

            fragments         = create_list_of_fragments(list_of_blobs.blobs_in_video, video.number_of_animals)
            list_of_fragments = ListOfFragments(fragments)

            video._fragment_identifier_to_index = list_of_fragments.get_fragment_identifier_to_index_list()
            global_fragments = create_list_of_global_fragments(
                list_of_blobs.blobs_in_video,
                list_of_fragments.fragments,
                video.number_of_animals
            )

            list_of_global_fragments = ListOfGlobalFragments(global_fragments)

            video.number_of_global_fragments = list_of_global_fragments.number_of_global_fragments
            list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()
            video.number_of_global_fragments_candidates_for_accumulation = list_of_global_fragments.number_of_global_fragments
            #XXX I skip the fit of the gamma ...
            list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
            video._number_of_unique_images_in_global_fragments = list_of_fragments.compute_total_number_of_images_in_global_fragments()
            list_of_global_fragments.compute_maximum_number_of_images()
            video._maximum_number_of_images_in_global_fragments = list_of_global_fragments.maximum_number_of_images
            list_of_fragments.get_accumulable_individual_fragments_identifiers(list_of_global_fragments)
            list_of_fragments.get_not_accumulable_individual_fragments_identifiers(list_of_global_fragments)
            list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
            list_of_fragments.save(video.fragments_path)
            list_of_fragments = list_of_fragments
        else:
            video._number_of_unique_images_in_global_fragments = None
            video._maximum_number_of_images_in_global_fragments = None
        video._has_been_preprocessed = True

        list_of_blobs.save(
            video, 
            video.blobs_path,
            number_of_chunks=video.number_of_frames
        )
        if video.number_of_animals != 1:
            list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
            list_of_global_fragments = list_of_global_fragments

        video.save()
        
        return crossing_detector_trainer, list_of_fragments, list_of_global_fragments










def step2_tracking(video):

    if video.number_of_animals == 1:

        #gui.tracker.track_single_animal()
        pass

    else:


        video.accumulation_trial = 0
        
        delete = True
        
        #if 'protocols1_and_2' in CHOSEN_VIDEO.processes_to_restore.keys():
        #    delete = not CHOSEN_VIDEO.processes_to_restore['protocols1_and_2']
        #else:
        #    delete = True
        
        video.create_accumulation_folder(iteration_number = 0, delete = delete)
        
        if not video.identity_transfer:
            n_animals = video.number_of_animals

        else:
            n_animals = video.knowledge_transfer_info_dict['number_of_animals']
        

        restoring_first_accumulation = False
        
        #self.init_accumulation_network()
        accumulation_network_params = NetworkParams(
            n_animals,
            learning_rate = 0.005,
            keep_prob     = 1.0,
            scopes_layers_to_optimize = None,
            save_folder = video.accumulation_folder,
            image_size  = video.identification_image_size,
            video_path  = video.video_path
        )




















def protocol1(video, list_of_fragments, list_of_global_fragments):

    if not video.identity_transfer:
        number_of_animals = video.number_of_animals

    else: 
        number_of_animals = video.knowledge_transfer_info_dict['number_of_animals']


    accumulation_network_params = NetworkParams(
        number_of_animals,
        learning_rate = 0.005,
        keep_prob = 1.0,
        scopes_layers_to_optimize = None,
        save_folder = video.accumulation_folder,
        image_size = video.identification_image_size,
        video_path = video.video_path
    )

    list_of_fragments.reset(roll_back_to = 'fragmentation')
    list_of_global_fragments.reset(roll_back_to = 'fragmentation')
    
    if video.tracking_with_knowledge_transfer:
        accumulation_network_params.scopes_layers_to_optimize = None
    
    net = ConvNetwork(accumulation_network_params)
    if video.tracking_with_knowledge_transfer:
        net.restore()
    
    video._first_frame_first_global_fragment.append(
        list_of_global_fragments.set_first_global_fragment_for_accumulation(
            video,
            accumulation_trial = 0
        )
    )

    if video.identity_transfer and\
       video.number_of_animals < video.knowledge_transfer_info_dict['number_of_animals']:
        tf.reset_default_graph()
        accumulation_network_params.number_of_animals = CHOSEN_VIDEO.video.number_of_animals
        accumulation_network_params._restore_folder = None
        accumulation_network_params.knowledge_transfer_folder = CHOSEN_VIDEO.video.knowledge_transfer_model_folder
        net = ConvNetwork(accumulation_network_params)
        net.restore()
    
    list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(video, accumulation_trial=0)
    
    accumulation_manager = AccumulationManager(
        video,
        list_of_fragments,
        list_of_global_fragments,
        threshold_acceptable_accumulation=THRESHOLD_ACCEPTABLE_ACCUMULATION
    )

    video.init_accumulation_statistics_attributes()
    accumulation_manager.threshold_early_stop_accumulation = THRESHOLD_EARLY_STOP_ACCUMULATION
    #self.global_step = 0
    #self.create_one_shot_accumulation_popup()
    #self.accumulation_step_finished = True
    #self.accumulation_loop()

    accumulation_step_finished = True

    if accumulation_step_finished and accumulation_manager.continue_accumulation:
        #self.one_shot_accumulation()
        
        accumulation_step_finished = False
        accumulation_manager.ratio_accumulated_images,\
        store_validation_accuracy_and_loss_data,\
        store_training_accuracy_and_loss_data = perform_one_accumulation_step(
            accumulation_manager,
            video,
            0,
            net,
            video.identity_transfer,
            save_summaries = False,#generate_tensorboard_switch.active,
            GUI_axes = None,
            net_properties = None,
            plot_flag = False
        )
        if accumulation_manager.counter == 1:
            #self.create_tracking_figures_axes()
            pass
        
    
    
    elif not accumulation_manager.continue_accumulation\
        and not video.first_accumulation_finished\
        and accumulation_manager.ratio_accumulated_images > THRESHOLD_EARLY_STOP_ACCUMULATION:
        
        video._first_accumulation_finished = True
        video._ratio_accumulated_images = accumulation_manager.ratio_accumulated_images
        video._percentage_of_accumulated_images = [video.ratio_accumulated_images]
        video.save()
        list_of_fragments.save(video.fragments_path)
        list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
        list_of_fragments.save_light_list(video._accumulation_folder)

        
    elif not self.accumulation_manager.continue_accumulation\
        and not CHOSEN_VIDEO.video.has_been_pretrained:
        
        video._first_accumulation_finished = True
        video._ratio_accumulated_images = accumulation_manager.ratio_accumulated_images
        video._percentage_of_accumulated_images = [video.ratio_accumulated_images]
        video.save()
        list_of_fragments.save(video.fragments_path)
        list_of_global_fragments.save(video.global_fragments_path, list_of_fragments.fragments)
        list_of_fragments.save_light_list(video._accumulation_folder)

        if accumulation_manager.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
            pass
        elif accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION:
            #self.protocol3()
            pass


    elif video.has_been_pretrained\
        and video.accumulation_trial < MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS\
        and accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION :
        
        video.accumulation_trial += 1
        if not accumulation_manager.continue_accumulation:

            video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
            video._percentage_of_accumulated_images.append(video.ratio_accumulated_images)
            list_of_fragments.save_light_list(video._accumulation_folder)

        #accumulation_parachute_init(video.accumulation_trial)
        #self.accumulation_loop()
    """elif CHOSEN_VIDEO.video.has_been_pretrained and\
        (self.accumulation_manager.ratio_accumulated_images >= THRESHOLD_ACCEPTABLE_ACCUMULATION\
        or CHOSEN_VIDEO.video.accumulation_trial >= MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS):
        Logger.info("Accumulation after protocol 3 has been successful")
        Logger.warning("************************ Unscheduling accumulate")
        Clock.unschedule(self.accumulate)
        Logger.warning("------------------------ dismissing one shot accumulation popup")
        self.one_shot_accumulation_popup.dismiss()
        self.save_after_second_accumulation()
        Logger.info("Start residual indentification")
        self.identification_popup.open()
    """




















