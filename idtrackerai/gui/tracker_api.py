import sys, time, os
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger as logger
else:
    import logging; logger = logging.getLogger(__name__)

import copy
import numpy as np

from idtrackerai.postprocessing.identify_non_assigned_with_interpolation import assign_zeros_with_interpolation_identities
from idtrackerai.postprocessing.correct_impossible_velocity_jumps        import correct_impossible_velocity_jumps
from idtrackerai.network.identification_model.network_params             import NetworkParams
from idtrackerai.postprocessing.compute_velocity_model                   import compute_model_velocity
from idtrackerai.network.identification_model.id_CNN                     import ConvNetwork
from idtrackerai.postprocessing.get_trajectories                         import produce_output_dict
from idtrackerai.postprocessing.assign_them_all                          import close_trajectories_gaps
from idtrackerai.accumulation_manager                                    import AccumulationManager
from idtrackerai.list_of_blobs                                           import ListOfBlobs
from idtrackerai.accumulator                                             import perform_one_accumulation_step
from idtrackerai.assigner                                                import assigner

from idtrackerai.constants import THRESHOLD_ACCEPTABLE_ACCUMULATION
from idtrackerai.constants import THRESHOLD_EARLY_STOP_ACCUMULATION
from idtrackerai.constants import VEL_PERCENTILE

class TrackerAPI(object):


    def __init__(self, chosen_video=None, **kwargs):

        self.chosen_video = chosen_video

        self.number_of_animals            = None # Number of animals
        self.accumulation_network_params  = None # Network params
        self.restoring_first_accumulation = False # Flag restores first accumulation
        self.accumulation_step_finished   = False # Flag accumulation step finished






    def init_tracking(self, gui_handler=None):

        if 'protocols1_and_2' in self.chosen_video.processes_to_restore.keys():
            delete = not self.chosen_video.processes_to_restore['protocols1_and_2']
        else:
            delete = True

        self.chosen_video.video.accumulation_trial = 0

        self.chosen_video.video.create_accumulation_folder(
            iteration_number = 0,
            delete = delete
        )

        if not self.chosen_video.video.identity_transfer:
            self.number_of_animals = self.chosen_video.video.number_of_animals
        else:
            self.number_of_animals = self.chosen_video.video.knowledge_transfer_info_dict['number_of_animals']

        self.restoring_first_accumulation = False

        self.accumulation_network_params = NetworkParams(
            self.number_of_animals,
            learning_rate = 0.005,
            keep_prob = 1.0,
            scopes_layers_to_optimize = None,
            save_folder = self.chosen_video.video.accumulation_folder,
            image_size = self.chosen_video.video.identification_image_size,
            video_path = self.chosen_video.video.video_path
        )

        # CHECK STATUS ###################################

        if self.status_post_processing: # POST PROCESSING

            self.restore_trajectories()
            self.restore_crossings_solved()
            self.restore_trajectories_wo_gaps()

            if gui_handler: gui_handler(0) # UPDATE GUI

        elif self.status_residual_identification: # RESIDUAL IDENTIFICATION

            if self.chosen_video.video.track_wo_identities: # TRACK WITHOUT IDENTITIES
                self.restore_trajectories()
                if gui_handler: gui_handler(1.1) # UPDATE GUI

            else: # TRACK WITH IDENTITIES
                logger.info("Restoring residual identification")
                self.restore_identification()
                self.chosen_video.video._has_been_assigned = True

                if gui_handler: gui_handler(1.2) # UPDATE GUI

        elif self.status_protocol3_accumulation: # PROTOCOL3 ACCUMULATION

            logger.info("Restoring second accumulation")
            self.restore_second_accumulation()
            self.chosen_video.video._first_frame_first_global_fragment = self.chosen_video.video._first_frame_first_global_fragment
            logger.warning(
                'first_frame_first_global_fragment {0}'.format(self.chosen_video.video.first_frame_first_global_fragment)
            )
            logger.info("Starting identification")

            if gui_handler: gui_handler(2) # UPDATE GUI

        elif self.status_protocol3_pretraining: # PROTOCOL3 PRETRAINNING

            logger.info("Restoring pretraining")
            logger.info("Initialising pretraining network")
            self.init_pretraining_net()
            logger.info("Restoring pretraining")
            self.accumulation_step_finished = True
            self.restore_first_accumulation()
            self.restore_pretraining()
            self.accumulation_manager.ratio_accumulated_images          = self.chosen_video.video.percentage_of_accumulated_images[0]
            self.chosen_video.video._first_frame_first_global_fragment  = [self.chosen_video.video._first_frame_first_global_fragment[0]]
            self.chosen_video.video._percentage_of_accumulated_images   = [self.chosen_video.video.percentage_of_accumulated_images[0]]
            logger.info("Start accumulation parachute")

            if gui_handler: gui_handler(3) # UPDATE GUI

        elif self.status_protocols1_and_2: # PROTOCOLS 1 AND 2

            logger.info("Restoring protocol 1 and 2")
            self.restoring_first_accumulation = True
            self.restore_first_accumulation()
            self.accumulation_manager.ratio_accumulated_images          = self.chosen_video.video.percentage_of_accumulated_images[0]
            self.chosen_video.video._first_frame_first_global_fragment  = [self.chosen_video.video._first_frame_first_global_fragment[0]]
            self.chosen_video.video._percentage_of_accumulated_images   = [self.chosen_video.video.percentage_of_accumulated_images[0]]
            self.accumulation_step_finished = True

            if gui_handler: gui_handler(4) # UPDATE GUI

        elif self.status_protocols1_and_2_not_def: # PROTOCOLS 1 AND 2 NOT DEFINED

            if gui_handler: gui_handler(5) # UPDATE GUI


    def track_single_animal(self):
        logger.debug("------------------------> track_single_animal")
        [setattr(b, '_identity', 1) for bf in self.chosen_video.list_of_blobs.blobs_in_video for b in bf]
        [setattr(b, '_P2_vector', [1.]) for bf in self.chosen_video.list_of_blobs.blobs_in_video for b in bf]
        [setattr(b, 'frame_number', frame_number) for frame_number, bf in enumerate(self.chosen_video.list_of_blobs.blobs_in_video) for b in bf]

    def track_single_global_fragment_video(self):
        logger.debug("------------------------> track_single_global_fragment_video")
        def get_P2_vector(identity, number_of_animals):
            P2_vector = np.zeros(number_of_animals)
            P2_vector[identity] = 1.
            return P2_vector
        [setattr(b, '_identity', b.fragment_identifier+1) for bf in self.chosen_video.list_of_blobs.blobs_in_video for b in bf]
        [setattr(b, '_P2_vector', get_P2_vector(b.fragment_identifier, self.chosen_video.video.number_of_animals))
            for bf in self.chosen_video.list_of_blobs.blobs_in_video for b in bf]
        self.chosen_video.video.accumulation_trial = 0
        self.chosen_video.video._first_frame_first_global_fragment = [0] # in case

    def track_wo_identities(self, *args):
        self.chosen_video.video.accumulation_trial = 0
        self.chosen_video.video._first_frame_first_global_fragment = [0]
        self.chosen_video.video._track_wo_identities = True
        



    def protocol1(self, create_popup=None):
        logger.debug("------------------------> protocol1")
        logger.debug("****** setting protocol1 time")
        self.chosen_video.video._protocol1_time = time.time()
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        self.chosen_video.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        # print("self.accumulation_network_params", self.accumulation_network_params.__dict__)
        # if self.chosen_video.video.tracking_with_knowledge_transfer:
        #     Logger.debug('Setting layers to optimize for knowledge_transfer')
        #     self.accumulation_network_params.scopes_layers_to_optimize = None #['fully-connected1','fully_connected_pre_softmax']
        self.net = ConvNetwork(self.accumulation_network_params)
        if self.chosen_video.video.tracking_with_knowledge_transfer:
            logger.debug('Restoring for knowledge transfer')
            self.net.restore()
        self.chosen_video.video._first_frame_first_global_fragment.append(self.chosen_video.list_of_global_fragments.set_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = 0))
        if self.chosen_video.video.identity_transfer and\
            self.chosen_video.video.number_of_animals < self.chosen_video.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = self.chosen_video.video.number_of_animals
            self.accumulation_network_params._restore_folder = None
            self.accumulation_network_params.knowledge_transfer_folder = self.chosen_video.video.knowledge_transfer_model_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
        self.chosen_video.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = 0)
        self.accumulation_manager = AccumulationManager(self.chosen_video.video, self.chosen_video.list_of_fragments,
                                                    self.chosen_video.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        self.global_step = 0

        if create_popup: create_popup()
        #self.create_one_shot_accumulation_popup()

        self.accumulation_step_finished = True
        self.accumulation_loop()


    def one_shot_accumulation(self, save_summaries=True):
        logger.debug("------------------------> one_shot_accumulation")
        logger.warning('Starting one_shot_accumulation')
        self.accumulation_step_finished = False
        self.accumulation_manager.ratio_accumulated_images,\
        self.store_validation_accuracy_and_loss_data,\
        self.store_training_accuracy_and_loss_data = perform_one_accumulation_step(
            self.accumulation_manager,
            self.chosen_video.video,
            self.global_step,
            self.net,
            self.chosen_video.video.identity_transfer,
            save_summaries = save_summaries,
            GUI_axes = None,
            net_properties = None,
            plot_flag = False
        )
        self.accumulation_step_finished = True




    def __accumulate_handler_call_accumulate(self):
        if self.accumulation_manager.continue_accumulation:
            self.accumulate()

    def __accumulate_handler_identification_popup_open(self):
        self.identify()
        self.postprocess_impossible_jumps()
        self.update_list_of_blobs()
        self.create_trajectories(interpolate_crossings_popup_open=self.interpolate_crossings)


    def accumulate(self,
            identification_popup_open           = None,
            one_shot_accumulation_popup_dismiss = None,
            create_pretraining_popup            = None,
            unschedule_accumulate               = None,
            call_accumulate                     = True
        ):

        if identification_popup_open is None: 
            identification_popup_open = self.__accumulate_handler_identification_popup_open

        logger.info("------------------------> Calling accumulate")

        if self.accumulation_step_finished and self.accumulation_manager.continue_accumulation:

            logger.info("--------------------> Performing accumulation")
            if self.accumulation_manager.counter == 1 and self.chosen_video.video.accumulation_trial == 0:
                self.chosen_video.video._protocol1_time = time.time()-self.chosen_video.video.protocol1_time
                self.chosen_video.video._protocol2_time = time.time()

            self.one_shot_accumulation()

        elif not self.accumulation_manager.continue_accumulation\
            and not self.chosen_video.video.first_accumulation_finished\
            and self.accumulation_manager.ratio_accumulated_images > THRESHOLD_EARLY_STOP_ACCUMULATION:

            logger.info("Protocol 1 successful")
            self.save_after_first_accumulation()
            if 'protocols1_and_2' not in self.chosen_video.processes_to_restore or not self.chosen_video.processes_to_restore['protocols1_and_2']:
                self.chosen_video.video._protocol1_time = time.time()-self.chosen_video.video.protocol1_time

            # call handler
            identification_popup_open()

        elif not self.accumulation_manager.continue_accumulation\
            and not self.chosen_video.video.has_been_pretrained:

            self.save_after_first_accumulation()

            if self.accumulation_manager.ratio_accumulated_images > THRESHOLD_ACCEPTABLE_ACCUMULATION:
                logger.info("Protocol 2 successful")
                logger.warning("------------------------ dismissing one shot accumulation popup")
                if one_shot_accumulation_popup_dismiss: one_shot_accumulation_popup_dismiss() # UPDATE GUI

                self.save_after_first_accumulation()
                if 'protocols1_and_2' not in self.chosen_video.processes_to_restore or not self.chosen_video.processes_to_restore['protocols1_and_2']:
                    self.chosen_video.video._protocol2_time = time.time()-self.chosen_video.video.protocol2_time

                # call handler
                identification_popup_open()

            elif self.accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION:

                logger.info("Protocol 2 failed -> Start protocol 3")
                if 'protocols1_and_2' not in self.chosen_video.processes_to_restore or not self.chosen_video.processes_to_restore['protocols1_and_2']:
                    self.chosen_video.video._protocol1_time = time.time()-self.chosen_video.video.protocol1_time
                    if self.chosen_video.video.protocol2_time != 0:
                        self.chosen_video.video._protocol2_time = time.time()-self.chosen_video.video.protocol2_time
                self.chosen_video.video._protocol3_pretraining_time = time.time()

                # call handler
                if create_pretraining_popup:
                    create_pretraining_popup()
                
                self.protocol3()

        elif self.chosen_video.video.has_been_pretrained\
            and self.chosen_video.video.accumulation_trial < MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS\
            and self.accumulation_manager.ratio_accumulated_images < THRESHOLD_ACCEPTABLE_ACCUMULATION:

            logger.info("Accumulation in protocol 3 is not successful. Opening parachute ...")
            if self.chosen_video.video.accumulation_trial == 0:
                self.chosen_video.video._protocol3_accumulation_time = time.time()
            self.chosen_video.video.accumulation_trial += 1
            if not self.accumulation_manager.continue_accumulation and self.chosen_video.video.accumulation_trial > 1:
                self.save_and_update_accumulation_parameters_in_parachute()
            self.accumulation_parachute_init(self.chosen_video.video.accumulation_trial)
            self.accumulation_loop()

        elif self.chosen_video.video.has_been_pretrained and\
            (self.accumulation_manager.ratio_accumulated_images >= THRESHOLD_ACCEPTABLE_ACCUMULATION\
            or self.chosen_video.video.accumulation_trial >= MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS):

            logger.info("Accumulation after protocol 3 has been successful")
            if 'protocol3_accumulation' not in self.chosen_video.processes_to_restore:
                self.chosen_video.video._protocol3_accumulation_time = time.time()-self.chosen_video.video.protocol3_accumulation_time
            elif 'protocol3_accumulation' in self.chosen_video.processes_to_restore and not self.chosen_video.processes_to_restore['protocol3_accumulation']:
                self.chosen_video.video._protocol3_accumulation_time = time.time()-self.chosen_video.video.protocol3_accumulation_time
            else:
                self.chosen_video.video._protocol3_accumulation_time = time.time()-self.chosen_video.video.protocol3_accumulation_time

            logger.warning("************************ Unscheduling accumulate")
            # call handler
            if unschedule_accumulate:
                unschedule_accumulate()
            
            logger.warning("------------------------ dismissing one shot accumulation popup")
            # call handler
            if one_shot_accumulation_popup_dismiss:
                one_shot_accumulation_popup_dismiss()
            
            self.save_after_second_accumulation()
            logger.info("Start residual indentification")
            # call handler
            identification_popup_open()

        # call handler
        if call_accumulate:
            self.__accumulate_handler_call_accumulate()
        

    def save_and_update_accumulation_parameters_in_parachute(self):
        logger.warning("self.accumulation_manager.ratio_accumulated_images %.4f" %self.accumulation_manager.ratio_accumulated_images)
        self.chosen_video.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
        self.chosen_video.video._percentage_of_accumulated_images.append(self.chosen_video.video.ratio_accumulated_images)
        self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)




    def accumulation_loop(self, do_accumulate=True):
        logger.warning('------------Calling accumulation loop')
        self.chosen_video.video.init_accumulation_statistics_attributes()
        self.accumulation_manager.threshold_early_stop_accumulation = THRESHOLD_EARLY_STOP_ACCUMULATION
        logger.warning('Calling accumulate from accumulation_loop')

        if do_accumulate: self.accumulate()


    def accumulation_parachute_init(self, 
        iteration_number,
        one_shot_accumulation_popup_dismiss=None):
        logger.debug("------------------------> accumulation_parachute_init")
        logger.info("Starting accumulation %i" %iteration_number)
        
        # Call to GUI
        if one_shot_accumulation_popup_dismiss:
            one_shot_accumulation_popup_dismiss()

        delete = not self.chosen_video.processes_to_restore['protocol3_accumulation'] if 'protocol3_accumulation' in self.chosen_video.processes_to_restore.keys() else True
        self.chosen_video.video.create_accumulation_folder(iteration_number = iteration_number, delete = delete)
        self.chosen_video.video.accumulation_trial = iteration_number
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        self.chosen_video.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        logger.info("We will restore the network from a previous pretraining: %s" %self.chosen_video.video.pretraining_folder)
        self.accumulation_network_params.save_folder = self.chosen_video.video.accumulation_folder
        self.accumulation_network_params.restore_folder = self.chosen_video.video.pretraining_folder
        self.accumulation_network_params.scopes_layers_to_optimize = ['fully-connected1','fully_connected_pre_softmax']
        logger.info("Initialising accumulation network")
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.net.reinitialize_softmax_and_fully_connected()
        logger.info("Initialising accumulation manager")
        self.chosen_video.video._first_frame_first_global_fragment.append(self.chosen_video.list_of_global_fragments.set_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = iteration_number - 1))
        logger.warning('first_frame_first_global_fragment ' + str(self.chosen_video.video.first_frame_first_global_fragment))
        if self.chosen_video.video.identity_transfer and self.chosen_video.video.number_of_animals < self.chosen_video.video.knowledge_transfer_info_dict['number_of_animals']:
            tf.reset_default_graph()
            self.accumulation_network_params.number_of_animals = self.chosen_video.video.number_of_animals
            self.accumulation_network_params.restore_folder = self.chosen_video.video.pretraining_folder
            self.net = ConvNetwork(self.accumulation_network_params)
            self.net.restore()
            self.net.reinitialize_softmax_and_fully_connected()
        self.chosen_video.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(self.chosen_video.video, accumulation_trial = iteration_number - 1)
        self.accumulation_manager = AccumulationManager(self.chosen_video.video,
                                                    self.chosen_video.list_of_fragments, self.chosen_video.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        logger.info("Start accumulation")
        self.global_step = 0


    def save_after_first_accumulation(self):
        """Set flags and save data"""
        logger.info("Saving first accumulation paramters")
        
        if not self.restoring_first_accumulation:
            self.chosen_video.video._first_accumulation_finished = True
            self.chosen_video.video._ratio_accumulated_images = self.accumulation_manager.ratio_accumulated_images
            self.chosen_video.video._percentage_of_accumulated_images = [self.chosen_video.video.ratio_accumulated_images]
            self.chosen_video.video._accumulation_network_params = self.accumulation_network_params
            self.chosen_video.video.save()
            self.chosen_video.list_of_fragments.save(self.chosen_video.video.fragments_path)
            self.chosen_video.list_of_global_fragments.save(self.chosen_video.video.global_fragments_path, self.chosen_video.list_of_fragments.fragments)
            self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)



    def save_after_second_accumulation(self):
        logger.info("Saving second accumulation parameters")
        self.save_and_update_accumulation_parameters_in_parachute()
        self.chosen_video.video.accumulation_trial = np.argmax(self.chosen_video.video.percentage_of_accumulated_images)
        self.chosen_video.video._ratio_accumulated_images = self.chosen_video.video.percentage_of_accumulated_images[self.chosen_video.video.accumulation_trial]
        accumulation_folder_name = 'accumulation_' + str(self.chosen_video.video.accumulation_trial)
        self.chosen_video.video._accumulation_folder = os.path.join(self.chosen_video.video.session_folder, accumulation_folder_name)
        self.chosen_video.list_of_fragments.load_light_list(self.chosen_video.video._accumulation_folder)
        self.chosen_video.video._second_accumulation_finished = True
        logger.info("Saving global fragments")
        self.chosen_video.list_of_fragments.save(self.chosen_video.video.fragments_path)
        self.chosen_video.list_of_global_fragments.save(self.chosen_video.video.global_fragments_path, self.chosen_video.list_of_fragments.fragments)
        logger.info("Restoring networks to best second accumulation")
        self.accumulation_network_params.restore_folder = self.chosen_video.video._accumulation_folder
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.chosen_video.video._accumulation_network_params = self.accumulation_network_params
        self.chosen_video.video.save()


    def init_pretraining_variables(self):
        self.init_pretraining_net()
        self.pretraining_global_step = 0
        self.net = ConvNetwork(self.pretrain_network_params)
        self.ratio_of_pretrained_images = 0
        if self.chosen_video.video.tracking_with_knowledge_transfer:
            self.net.restore()
        self.store_training_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'training',
                                                                                    scope = 'pretraining')
        self.store_validation_accuracy_and_loss_data_pretrain = Store_Accuracy_and_Loss(self.net,
                                                                                    name = 'validation',
                                                                                    scope = 'pretraining')

    def pretraining_loop(self, do_continue_pretraining=True):
        logger.debug("------------------------> pretraining_loop")
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        self.chosen_video.list_of_global_fragments.order_by_distance_travelled()
        
        ## IT SHOULD NOT BE CALLED BY THE G
        if do_continue_pretraining:
            self.continue_pretraining()
        
    def continue_pretraining(self, *args, clock_unschedule=None):
        logger.debug("------------------------> continue_pretraining")
        if self.pretraining_step_finished and self.ratio_of_pretrained_images < MAX_RATIO_OF_PRETRAINED_IMAGES:
            self.one_shot_pretraining()
        elif self.ratio_of_pretrained_images > MAX_RATIO_OF_PRETRAINED_IMAGES:
            self.chosen_video.video._has_been_pretrained = True
            
            # Call GUI
            if clock_unschedule: clock_unschedule()

            logger.warning('Calling accumulate from continue_pretraining')
            logger.debug('****** saving protocol3 pretraining time')
            self.chosen_video.video._protocol3_pretraining_time = time.time()-self.chosen_video.video.protocol3_pretraining_time
            self.accumulate()

    def one_shot_pretraining(self, *args):
        logger.debug("------------------------> one_shot_pretraining")
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = self.chosen_video.list_of_global_fragments.global_fragments[self.pretraining_counter]
        self.net,\
        self.ratio_of_pretrained_images,\
        pretraining_global_step,\
        self.store_training_accuracy_and_loss_data_pretrain,\
        self.store_validation_accuracy_and_loss_data_pretrain,\
        self.chosen_video.list_of_fragments = pre_train_global_fragment(self.net,
                                                    self.pretraining_global_fragment,
                                                    self.chosen_video.list_of_fragments,
                                                    self.pretraining_global_step,
                                                    True, True,
                                                    self.generate_tensorboard_switch.active,
                                                    self.store_training_accuracy_and_loss_data_pretrain,
                                                    self.store_validation_accuracy_and_loss_data_pretrain,
                                                    print_flag = False,
                                                    plot_flag = False,
                                                    batch_size = BATCH_SIZE_IDCNN,
                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.pretraining_counter += 1
        self.pretraining_counter_value.text = str(self.pretraining_counter)
        self.percentage_pretrained_images_value.text = str(self.ratio_of_pretrained_images)
        self.store_training_accuracy_and_loss_data_pretrain.plot_global_fragments(self.pretrain_ax_arr,
                                                                    self.chosen_video.video,
                                                                    self.chosen_video.list_of_fragments.fragments,
                                                                    black = False,
                                                                    canvas_from_GUI = self.pretrain_fig_canvas)
        self.store_validation_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color ='b',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    index = self.pretraining_global_step,
                                                    legend_font_color = 'w')
        self.store_training_accuracy_and_loss_data_pretrain.plot(self.pretrain_ax_arr,
                                                    color = 'r',
                                                    canvas_from_GUI = self.pretrain_fig_canvas,
                                                    index = self.pretraining_global_step,
                                                    legend_font_color = 'w')
        self.pretraining_step_finished = True

    def protocol3(self):
        logger.debug("------------------------> protocol3")
        self.init_pretraining_variables()
        number_of_images_in_global_fragments = self.chosen_video.video.number_of_unique_images_in_global_fragments
        if self.chosen_video.old_video and self.chosen_video.old_video.first_accumulation_finished == True:
            self.chosen_video.list_of_global_fragments.reset(roll_back_to = 'fragmentation')
            self.chosen_video.list_of_fragments.reset(roll_back_to = 'fragmentation')
        logger.info("Starting pretraining. Checkpoints will be stored in %s" %self.chosen_video.video.pretraining_folder)
        if self.chosen_video.video.tracking_with_knowledge_transfer:
            logger.info("Performing knowledge transfer from %s" %self.chosen_video.video.knowledge_transfer_model_folder)
            self.pretrain_network_params.knowledge_transfer_folder = self.chosen_video.video.knowledge_transfer_model_folder
        logger.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()






    def init_pretraining_net(self):
        delete = not self.chosen_video.processes_to_restore['protocol3_pretraining'] if 'protocol3_pretraining' in self.chosen_video.processes_to_restore.keys() else True
        self.chosen_video.video.create_pretraining_folder(delete = delete)
        self.pretrain_network_params = NetworkParams(self.chosen_video.video.number_of_animals,
                                                learning_rate = 0.01,
                                                keep_prob = 1.0,
                                                use_adam_optimiser = False,
                                                scopes_layers_to_optimize = None,
                                                save_folder = self.chosen_video.video.pretraining_folder,
                                                image_size = self.chosen_video.video.identification_image_size,
                                                video_path = self.chosen_video.video.video_path)
        self.chosen_video.video._pretraining_network_params = self.pretrain_network_params



    def restore_video_attributes(self):
        list_of_attributes = ['accumulation_folder',
                    'second_accumulation_finished',
                    'number_of_accumulated_global_fragments',
                    'number_of_non_certain_global_fragments',
                    'number_of_randomly_assigned_global_fragments',
                    'number_of_nonconsistent_global_fragments',
                    'number_of_nonunique_global_fragments',
                    'number_of_acceptable_global_fragments',
                    'validation_accuracy', 'validation_individual_accuracies',
                    'training_accuracy', 'training_individual_accuracies',
                    'percentage_of_accumulated_images', 'accumulation_trial',
                    'ratio_accumulated_images', 'first_accumulation_finished',
                    'identity_transfer', 'accumulation_statistics',
                    'first_frame_first_global_fragment', 'pretraining_folder',
                    'has_been_pretrained', 'has_been_assigned',
                    'has_crossings_solved','has_trajectories',
                    'has_trajectories_wo_gaps',
                    'protocol1_time', 'protocol2_time',
                    'protocol3_pretraining_time', 'protocol3_accumulation_time',
                    'identify_time', 'create_trajectories_time']
        is_property = [True, True, False, False, False, False, False, False,
                        False, False, False, False, True, False, True, True,
                        True, False, True, True, True, True, True, True, True,
                        True, True, True, True, True, True]
        self.chosen_video.video.copy_attributes_between_two_video_objects(self.chosen_video.old_video, list_of_attributes, is_property = is_property)

    def restore_first_accumulation(self):
        self.restore_video_attributes()
        self.chosen_video.video._ratio_accumulated_images = self.chosen_video.video.percentage_of_accumulated_images[0]
        self.accumulation_network_params.restore_folder = self.chosen_video.video._accumulation_folder
        self.accumulation_manager = AccumulationManager(self.chosen_video.video, self.chosen_video.list_of_fragments,
                                                    self.chosen_video.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        logger.info("Saving video")
        self.chosen_video.video._has_been_pretrained = False
        self.chosen_video.video.save()
        self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)

    def restore_pretraining(self):
        logger.info("Restoring pretrained network")
        self.restore_video_attributes()
        self.pretrain_network_params.restore_folder = self.chosen_video.video.pretraining_folder
        self.net = ConvNetwork(self.pretrain_network_params)
        self.net.restore()
        self.accumulation_manager = AccumulationManager(self.chosen_video.video, self.chosen_video.list_of_fragments,
                                                    self.chosen_video.list_of_global_fragments,
                                                    threshold_acceptable_accumulation = THRESHOLD_ACCEPTABLE_ACCUMULATION)
        self.chosen_video.video.accumulation_trial = 0
        self.chosen_video.video.save()

    def restore_second_accumulation(self):
        self.restore_video_attributes()
        logger.info("Restoring trained network")
        self.accumulation_network_params.restore_folder = self.chosen_video.video._accumulation_folder
        self.chosen_video.list_of_fragments.load_light_list(self.chosen_video.video._accumulation_folder)
        self.net = ConvNetwork(self.accumulation_network_params)
        self.net.restore()
        self.chosen_video.video.save()

    def restore_identification(self):
        self.restore_video_attributes()
        self.chosen_video.list_of_fragments.load_light_list(self.chosen_video.video._accumulation_folder)
        self.chosen_video.video.save()

    def restore_trajectories(self):
        self.restore_video_attributes()
        self.chosen_video.video.save()

    def restore_crossings_solved(self):
        self.restore_video_attributes()
        self.chosen_video.video.copy_attributes_between_two_video_objects(self.chosen_video.old_video, ['blobs_no_gaps_path'], [False])
        self.chosen_video.list_of_blobs_no_gaps = ListOfBlobs.load(self.chosen_video.video, self.chosen_video.video.blobs_no_gaps_path)
        self.chosen_video.video.save()

    def restore_trajectories_wo_gaps(self):
        self.restore_video_attributes()
        self.chosen_video.video.save()

    def identify(self, *args):
        self.chosen_video.video._identify_time = time.time()
        logger.warning("In identify")
        self.chosen_video.list_of_fragments.reset(roll_back_to = 'accumulation')
        logger.warning("Calling assigner")
        assigner(self.chosen_video.list_of_fragments, self.chosen_video.video, self.net)
        self.chosen_video.video._has_been_assigned = True
        self.chosen_video.video.save()


    def postprocess_impossible_jumps(self, *args):
        if not hasattr(self.chosen_video.video, 'velocity_threshold') and hasattr(self.chosen_video.old_video,'velocity_threshold'):
            self.chosen_video.video.velocity_threshold = self.chosen_video.old_video.velocity_threshold
        elif not hasattr(self.chosen_video.old_video, 'velocity_threshold'):
            self.chosen_video.video.velocity_threshold = compute_model_velocity(
                                                                self.chosen_video.list_of_fragments.fragments,
                                                                self.chosen_video.video.number_of_animals,
                                                                percentile = VEL_PERCENTILE)
        correct_impossible_velocity_jumps(self.chosen_video.video, self.chosen_video.list_of_fragments)
        self.chosen_video.list_of_fragments.save(self.chosen_video.video.fragments_path)
        self.chosen_video.video.save()
        

    def update_list_of_blobs(self, *args):
        self.chosen_video.video.individual_fragments_stats = self.chosen_video.list_of_fragments.get_stats(self.chosen_video.list_of_global_fragments)
        self.chosen_video.video.compute_overall_P2(self.chosen_video.list_of_fragments.fragments)
        self.chosen_video.list_of_fragments.save_light_list(self.chosen_video.video._accumulation_folder)
        self.chosen_video.video.save()
        if not hasattr(self.chosen_video, 'list_of_blobs'):
            self.chosen_video.list_of_blobs = ListOfBlobs.load(self.chosen_video.video, self.chosen_video.old_video.blobs_path)
        self.chosen_video.list_of_blobs.update_from_list_of_fragments(self.chosen_video.list_of_fragments.fragments,
                                                    self.chosen_video.video.fragment_identifier_to_index)
        # if False:
        #     self.chosen_video.list_of_blobs.compute_nose_and_head_coordinates()
        self.chosen_video.list_of_blobs.save(self.chosen_video.video,
                                        self.chosen_video.video.blobs_path,
                                        number_of_chunks = self.chosen_video.video.number_of_frames)
        self.chosen_video.video._identify_time = time.time()-self.chosen_video.video.identify_time


    def create_trajectories(self, *args, 
        trajectories_popup_dismiss=None,
        interpolate_crossings_popup_open=None,
        update_and_show_happy_ending_popup=None ):

        self.chosen_video.video._create_trajectories_time = time.time()
        if 'post_processing' not in self.chosen_video.processes_to_restore or not self.chosen_video.processes_to_restore['post_processing']:
            if not self.chosen_video.video.track_wo_identities:
                self.chosen_video.video.create_trajectories_folder()
                trajectories_file = os.path.join(self.chosen_video.video.trajectories_folder, 'trajectories.npy')
                trajectories = produce_output_dict(self.chosen_video.list_of_blobs.blobs_in_video, self.chosen_video.video)
            else:
                self.chosen_video.video.create_trajectories_wo_identities_folder()
                trajectories_file = os.path.join(self.chosen_video.video.trajectories_wo_identities_folder, 'trajectories_wo_identities.npy')
                trajectories = produce_output_dict(self.chosen_video.list_of_blobs.blobs_in_video, self.chosen_video.video)
            np.save(trajectories_file, trajectories)
            logger.info("Saving trajectories")
            np.save(trajectories_file, trajectories)
        self.chosen_video.video._has_trajectories = True
        self.chosen_video.video.save()

        # Call GUI function
        if trajectories_popup_dismiss: trajectories_popup_dismiss()


        if self.chosen_video.video.number_of_animals != 1 and self.chosen_video.list_of_global_fragments.number_of_global_fragments != 1 and not self.chosen_video.video.track_wo_identities:
            # Call GUI function
            if interpolate_crossings_popup_open: interpolate_crossings_popup_open()
        else:
            self.chosen_video.video.overall_P2 = 1.
            self.chosen_video.video._has_been_assigned = True
            self.chosen_video.video._has_crossings_solved = False
            self.chosen_video.video._has_trajectories_wo_gaps = False
            self.chosen_video.list_of_blobs.save(self.chosen_video.video,
                                            self.chosen_video.video.blobs_path,
                                            number_of_chunks = self.chosen_video.video.number_of_frames)
            # Call GUI function
            if update_and_show_happy_ending_popup: update_and_show_happy_ending_popup()



    def interpolate_crossings(self, *args):
        self.chosen_video.list_of_blobs_no_gaps = copy.deepcopy(self.chosen_video.list_of_blobs)
        # if not hasattr(self.chosen_video.list_of_blobs_no_gaps.blobs_in_video[0][0], '_was_a_crossing'):
        #     Logger.debug("adding attribute was_a_crossing to every blob")
        #     [setattr(blob, '_was_a_crossing', False) for blobs_in_frame in
        #         self.chosen_video.list_of_blobs_no_gaps.blobs_in_video for blob in blobs_in_frame]
        self.chosen_video.video._has_crossings_solved = False
        self.chosen_video.list_of_blobs_no_gaps = close_trajectories_gaps(self.chosen_video.video, self.chosen_video.list_of_blobs_no_gaps, self.chosen_video.list_of_fragments)
        self.chosen_video.video.blobs_no_gaps_path = os.path.join(os.path.split(self.chosen_video.video.blobs_path)[0], 'blobs_collection_no_gaps.npy')
        self.chosen_video.list_of_blobs_no_gaps.save(self.chosen_video.video, path_to_save = self.chosen_video.video.blobs_no_gaps_path, number_of_chunks = self.chosen_video.video.number_of_frames)
        self.chosen_video.video._has_crossings_solved = True
        self.chosen_video.video.save()
        

    def create_trajectories_wo_gaps(self, *args):
        self.chosen_video.video.create_trajectories_wo_gaps_folder()
        logger.info("Generating trajectories. The trajectories files are stored in %s" %self.chosen_video.video.trajectories_wo_gaps_folder)
        trajectories_wo_gaps_file = os.path.join(self.chosen_video.video.trajectories_wo_gaps_folder, 'trajectories_wo_gaps.npy')
        trajectories_wo_gaps = produce_output_dict(self.chosen_video.list_of_blobs_no_gaps.blobs_in_video, self.chosen_video.video)
        np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
        self.chosen_video.video._has_trajectories_wo_gaps = True
        logger.info("Saving trajectories")
        self.chosen_video.list_of_blobs = assign_zeros_with_interpolation_identities(self.chosen_video.list_of_blobs, self.chosen_video.list_of_blobs_no_gaps)
        trajectories_file = os.path.join(self.chosen_video.video.trajectories_folder, 'trajectories.npy')
        trajectories = produce_output_dict(self.chosen_video.list_of_blobs.blobs_in_video, self.chosen_video.video)
        np.save(trajectories_file, trajectories)
        self.chosen_video.video.save()
        self.chosen_video.video._create_trajectories_time = time.time()-self.chosen_video.video.create_trajectories_time
        
    def update_and_show_happy_ending_popup(self, *args):
        if not hasattr(self.chosen_video.video, 'overall_P2'):
            self.chosen_video.video.compute_overall_P2(self.chosen_video.list_of_fragments.fragments)
        self.chosen_video.video.save()
        
    def network_params_to_string(self):
        self.str_model = str(self.accumulation_network_params.cnn_model)
        self.str_lr = str(self.accumulation_network_params.learning_rate)
        self.str_kp = str(self.accumulation_network_params.keep_prob)
        self.str_optimiser = "SGD" if not self.accumulation_network_params.use_adam_optimiser else "Adam"
        self.str_layers_to_train = "all" if self.accumulation_network_params.scopes_layers_to_optimize is None else str(self.accumulation_network_params.scopes_layers_to_optimize)
        self.restore_folder = self.accumulation_network_params.restore_folder if self.accumulation_network_params.restore_folder is not None else 'None'
        self.save_folder = self.accumulation_network_params.save_folder if self.accumulation_network_params.save_folder is not None else 'None'
        self.knowledge_transfer_folder = self.accumulation_network_params.knowledge_transfer_folder if self.accumulation_network_params.knowledge_transfer_folder is not None else 'None'
        # self.kt_conv_layers_to_discard = self.accumulation_network_params.kt_conv_layers_to_discard if self.accumulation_network_params.kt_conv_layers_to_discard is not None else 'None'

    ############################################################################################
    ### PROPERTIES #############################################################################
    ############################################################################################

    ### STATUS ###

    @property
    def status_post_processing(self):
        #return 'post_processing' in self.chosen_video.processes_to_restore and self.chosen_video.processes_to_restore['post_processing']
        return self.chosen_video.processes_to_restore.get('post_processing', None) is not None

    @property
    def status_residual_identification(self):
        #return 'residual_identification' in self.chosen_video.processes_to_restore and self.chosen_video.processes_to_restore['residual_identification']
        return self.chosen_video.processes_to_restore.get('residual_identification', None) is not None

    @property
    def status_protocol3_accumulation(self):
        #return 'protocol3_accumulation' in self.chosen_video.processes_to_restore and self.chosen_video.processes_to_restore['protocol3_accumulation']
        return self.chosen_video.processes_to_restore.get('protocol3_accumulation', None) is not None

    @property
    def status_protocol3_pretraining(self):
        #return 'protocol3_pretraining' in self.chosen_video.processes_to_restore and self.chosen_video.processes_to_restore['protocol3_pretraining']
        return self.chosen_video.processes_to_restore.get('protocol3_pretraining', None) is not None

    @property
    def status_protocols1_and_2(self):
        #return 'protocols1_and_2' in self.chosen_video.processes_to_restore and self.chosen_video.processes_to_restore['protocols1_and_2']
        return self.chosen_video.processes_to_restore.get('protocols1_and_2', None) is not None

    @property
    def status_protocols1_and_2_not_def(self):
        #return 'protocols1_and_2' not in self.chosen_video.processes_to_restore or not self.chosen_video.processes_to_restore['protocols1_and_2']
        return self.chosen_video.processes_to_restore.get('protocols1_and_2', None) is None
