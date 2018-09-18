from kivy.logger import Logger as logger

from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.network.identification_model.id_CNN         import ConvNetwork
from idtrackerai.accumulation_manager                        import AccumulationManager
from idtrackerai.list_of_blobs                               import ListOfBlobs

class TrackerAPI(object):


    def __init__(self, chosen_video=None, **kwargs):
        
        self.chosen_video = chosen_video
        
        self.number_of_animals            = None # Number of animals
        self.accumulation_network_params  = None # Network params
        self.restoring_first_accumulation = False # Flag restores first accumulation
        self.accumulation_step_finished   = False # Flag accumulation step finished






    def init_tracking(self, update_gui=None):

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

            if update_gui: update_gui(0) # UPDATE GUI
        
        elif self.status_residual_identification: # RESIDUAL IDENTIFICATION
            
            if self.chosen_video.video.track_wo_identities: # TRACK WITHOUT IDENTITIES
                self.restore_trajectories()
                if update_gui: update_gui(1.1) # UPDATE GUI

            else: # TRACK WITH IDENTITIES
                logger.info("Restoring residual identification")
                self.restore_identification()
                self.chosen_video.video._has_been_assigned = True
                
                if update_gui: update_gui(1.2) # UPDATE GUI

        elif self.status_protocol3_accumulation: # PROTOCOL3 ACCUMULATION

            logger.info("Restoring second accumulation")
            self.restore_second_accumulation()
            self.chosen_video.video._first_frame_first_global_fragment = self.chosen_video.video._first_frame_first_global_fragment
            logger.warning(
                'first_frame_first_global_fragment {0}'.format(self.chosen_video.video.first_frame_first_global_fragment)
            )
            logger.info("Starting identification")

            if update_gui: update_gui(2) # UPDATE GUI

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

            if update_gui: update_gui(3) # UPDATE GUI
        
        elif self.status_protocols1_and_2: # PROTOCOLS 1 AND 2

            logger.info("Restoring protocol 1 and 2")
            self.restoring_first_accumulation = True
            self.restore_first_accumulation()
            self.accumulation_manager.ratio_accumulated_images          = self.chosen_video.video.percentage_of_accumulated_images[0]
            self.chosen_video.video._first_frame_first_global_fragment  = [self.chosen_video.video._first_frame_first_global_fragment[0]]
            self.chosen_video.video._percentage_of_accumulated_images   = [self.chosen_video.video.percentage_of_accumulated_images[0]]
            self.accumulation_step_finished = True

            if update_gui: update_gui(4) # UPDATE GUI

        elif self.status_protocols1_and_2_not_def: # PROTOCOLS 1 AND 2 NOT DEFINED

            if update_gui: update_gui(5) # UPDATE GUI







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
    