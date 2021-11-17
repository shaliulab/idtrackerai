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

import copy
import logging
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from confapp import conf

from idtrackerai.tracker.accumulation_manager import AccumulationManager
from idtrackerai.tracker.accumulator import perform_one_accumulation_step
from idtrackerai.tracker.assigner import assign_remaining_fragments
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.list_of_global_fragments import (
    ListOfGlobalFragments,
    create_list_of_global_fragments,
)
from idtrackerai.tracker.network.network_params import (
    NetworkParams,
)
from idtrackerai.network.learners.learners import Learner_Classification
from idtrackerai.tracker.assign_them_all import (
    close_trajectories_gaps,
)
from idtrackerai.tracker.compute_velocity_model import (
    compute_model_velocity,
)

# from idtrackerai.network.identification_model.store_accuracy_and_loss import Store_Accuracy_and_Loss
from idtrackerai.tracker.correct_impossible_velocity_jumps import (
    correct_impossible_velocity_jumps,
)

# from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.tracker.get_trajectories import (
    produce_output_dict,
)
from idtrackerai.tracker.identify_non_assigned_with_interpolation import (
    assign_zeros_with_interpolation_identities,
)
from idtrackerai.tracker.trajectories_to_csv import (
    convert_trajectories_file_to_csv_and_json,
)
from idtrackerai.tracker.pre_trainer import pre_train_global_fragment
from idtrackerai.network.utils.utils import (
    fc_weights_reinit,
    weights_xavier_init,
)

logger = logging.getLogger(__name__)


class TrackerAPI(object):
    def __init__(self, video, list_of_blobs, list_of_fragments):

        self.video = video
        self.list_of_blobs = list_of_blobs
        self.list_of_fragments = list_of_fragments
        self.list_of_global_fragments = None

        if (
            self.video.user_defined_parameters["knowledge_transfer_folder"]
            is not None
        ):
            kt_info_dict_path = os.path.join(
                self.video.user_defined_parameters[
                    "knowledge_transfer_folder"
                ],
                "model_params.npy",
            )
            assert os.path.exists(kt_info_dict_path)
            self.knowledge_transfer_info_dict = np.load(
                kt_info_dict_path, allow_pickle=True
            ).item()
        else:
            self.knowledge_transfer_info_dict = None

        # Old requirements for restoring
        self.processes_to_restore = {}

        self.number_of_identities = None  # Number of identities
        self.accumulation_network_params = None  # Network params
        self.restoring_first_accumulation = (
            False  # Flag restores first accumulation
        )
        self.accumulation_step_finished = (
            False  # Flag accumulation step finished
        )

    @staticmethod
    def check_if_identity_transfer_is_possible(
        number_of_animals,
        knowledge_transfer_folder,
    ):
        if knowledge_transfer_folder is None:
            raise ValueError(
                "To perform identity transfer you "
                "need to provide a path for the variable "
                "KNOWLEDGE_TRANSFER_FOLDER_IDCNN "
                "in the local_settings.py file"
            )

        kt_info_dict_path = os.path.join(
            knowledge_transfer_folder, "model_params.npy"
        )
        if os.path.isfile(kt_info_dict_path):
            knowledge_transfer_info_dict = np.load(
                kt_info_dict_path, allow_pickle=True
            ).item()
            assert "image_size" in knowledge_transfer_info_dict
        else:
            raise ValueError(
                "To perform identity transfer the models_params.npy file "
                "is needed to check the "
                "input_image_size and the number_of_classes of the model to "
                "be loaded"
            )
        is_identity_transfer_possible = (
            number_of_animals
            == knowledge_transfer_info_dict["number_of_classes"]
        )
        if is_identity_transfer_possible:
            logger.info(
                "Tracking with identity transfer. "
                "The identification_image_size will be matched "
                "to the image_size of the transferred network"
            )
            identification_image_size = knowledge_transfer_info_dict[
                "image_size"
            ]
        else:
            logger.warning(
                "Tracking with identity transfer is not possible. "
                "The number of animals in the video needs to be the same as "
                "the number of animals in the transferred network"
            )
            identification_image_size = None

        return is_identity_transfer_possible, identification_image_size

    def track_single_animal(self, create_trajectories=None):

        if create_trajectories is None:
            create_trajectories = self.create_trajectories

        logger.debug("---> track_single_animal")
        for f, bf in enumerate(self.list_of_blobs.blobs_in_video):
            for blob in bf:
                blob._identity = 1
                blob._P2_vector = [1.0]
                blob.frame_number = f

        create_trajectories()

    def _get_global_fragments(self):
        # GLOBAL FRAGMENTS
        global_fragments = create_list_of_global_fragments(
            self.list_of_blobs.blobs_in_video,
            self.list_of_fragments.fragments,
            self.video.user_defined_parameters["number_of_animals"],
        )
        # Create list of global fragments
        list_of_global_fragments = ListOfGlobalFragments(global_fragments)
        self._other_operation_with_fragments_and_global_fragments(
            self.list_of_fragments, list_of_global_fragments
        )
        return list_of_global_fragments

    def _other_operation_with_fragments_and_global_fragments(
        self, list_of_fragments, list_of_global_fragments
    ):
        # Filter candidates global fragments for accumulation
        list_of_global_fragments.filter_candidates_global_fragments_for_accumulation()

        list_of_global_fragments.relink_fragments_to_global_fragments(
            list_of_fragments.fragments
        )
        list_of_global_fragments.compute_maximum_number_of_images()

        list_of_fragments.get_accumulable_individual_fragments_identifiers(
            list_of_global_fragments
        )
        list_of_fragments.get_not_accumulable_individual_fragments_identifiers(
            list_of_global_fragments
        )
        list_of_fragments.set_fragments_as_accumulable_or_not_accumulable()
        list_of_fragments.compute_total_number_of_images_in_global_fragments()

    def track_multiple_animals(self):
        list_of_global_fragments = self._get_global_fragments()
        self.list_of_global_fragments = list_of_global_fragments
        if list_of_global_fragments.number_of_global_fragments == 1:
            logger.info("START: TRACKING SINGLE GLOBAL FRAGMENT")
            self._track_single_global_fragment_video()
            self.list_of_fragments.save(self.video.fragments_path)
            self.list_of_global_fragments.save(
                self.video.global_fragments_path,
                self.list_of_fragments.fragments,
            )
            logger.info("FINISH: TRACKING SINGLE GLOBAL FRAGMENT")

        else:
            logger.info("START: TRACKING")
            self._track_w_identities()
            logger.info("FINISH: TRACKING")

    def _track_single_global_fragment_video(self, create_trajectories=None):
        def get_P2_vector(identity, number_of_animals):
            P2_vector = np.zeros(number_of_animals)
            P2_vector[identity - 1] = 1.0
            return P2_vector

        if create_trajectories is None:
            create_trajectories = self.create_trajectories

        logger.debug("---> track_single_global_fragment_video")

        fragment_identifier_to_id = {}
        identity = 1
        for fragment in self.list_of_fragments.fragments:
            if fragment.is_an_individual:
                fragment_identifier_to_id[fragment.identifier] = identity
                identity += 1
            else:
                fragment_identifier_to_id[fragment.identifier] = None

        for f, bf in enumerate(self.list_of_blobs.blobs_in_video):
            for b in bf:
                if b.is_an_individual:
                    b._identity = fragment_identifier_to_id[
                        b.fragment_identifier
                    ]
                    b._P2_vector = get_P2_vector(
                        fragment_identifier_to_id[b.fragment_identifier],
                        self.video.user_defined_parameters[
                            "number_of_animals"
                        ],
                    )
                    b.frame_number = f
        self.video._first_frame_first_global_fragment = [0]  # in case
        create_trajectories()

    def track_wo_identification(self, create_trajectories=None):

        if create_trajectories is None:
            create_trajectories = self.create_trajectories

        self.video._first_frame_first_global_fragment = [0]
        self.video._track_wo_identification = True
        create_trajectories()

    def _track_w_identities(self):
        track_with_cascade = True
        if track_with_cascade:
            # This runs the protocol cascade and also the residual
            # identification, the impossible_jumps, the creation of
            # trajectories, the crossings interpolation, and the
            # creation of trajectories_wo_gaps
            # TODO: Factorize track_with_protocols_cascade so it only runs
            # up to residual identification
            self._track_with_protocols_cascade()
        else:
            # TODO: Here is where new tracking methods should come
            # Call to tracking method

            # Call to postprocessing
            # TODO: Factorize postprocess_impossible_jumps
            # postprocess_impossible_jumps
            # create_trajectories
            # crossings_interpolation
            # create_trajectories_wo_gaps
            self.postprocess_impossible_jumps()
            raise NotImplementedError("New tracking methods are not allwoed")

    def _track_with_protocols_cascade(self):
        logger.info("******* Start tracking with protocol cascade ********")
        # Restoring
        if "protocols1_and_2" in self.processes_to_restore:
            delete = not self.processes_to_restore["protocols1_and_2"]
        else:
            delete = True

        # Create accumulation folder
        self.video.create_accumulation_folder(
            iteration_number=0, delete=delete
        )

        # Set number of animals params for identity transfer
        if not self.video.user_defined_parameters["identity_transfer"]:
            self.number_of_identities = self.video.user_defined_parameters[
                "number_of_animals"
            ]
        else:
            self.number_of_identities = self.knowledge_transfer_info_dict[
                "number_of_classes"
            ]

        self.init_accumulation_idCNN_params()

        # Restoring
        self.restoring_first_accumulation = False
        if (
            "post_processing" in self.processes_to_restore
            and self.processes_to_restore["post_processing"]
        ):
            raise
            # self.restore_trajectories()
            # self.restore_crossings_solved()
            # self.restore_trajectories_wo_gaps()

        elif (
            "residual_identification" in self.processes_to_restore
            and self.processes_to_restore["residual_identification"]
        ):
            if self.video.user_defined_parameters["track_wo_identification"]:
                # TODO: bring restoring back to life
                raise
                # self.restore_trajectories()

            else:
                # TODO: bring restoring back to life
                raise
                # logger.info("Restoring residual identification")
                # self.restore_identification()
                # self.create_trajectories()

        elif (
            "protocol3_accumulation" in self.processes_to_restore
            and self.processes_to_restore["protocol3_accumulation"]
        ):
            raise
            # logger.info("Restoring second accumulation")
            # # self.restore_second_accumulation()
            # self.video._first_frame_first_global_fragment = (
            #     self.video._first_frame_first_global_fragment
            # )
            # logger.warning(
            #     "first_frame_first_global_fragment "
            #     + str(
            #         self.video.first_frame_first_global_fragment
            #     )
            # )
            # logger.info("Starting identification")
            #
            # self.create_trajectories()

        elif (
            "protocol3_pretraining" in self.processes_to_restore
            and self.processes_to_restore["protocol3_pretraining"]
        ):
            # TODO: bring restoring back to life
            raise
            # logger.info("Restoring pretraining")
            # logger.info("Initialising pretraining network")
            # self.init_pretraining_net()
            # logger.info("Restoring pretraining")
            # self.accumulation_step_finished = True
            # self.restore_first_accumulation()
            # self.restore_pretraining()
            # self.accumulation_manager.ratio_accumulated_images =
            # self.video.percentage_of_accumulated_images[0]
            # self.video._first_frame_first_global_fragment = [
            #     self.video._first_frame_first_global_fragment[
            #         0
            #     ]
            # ]
            # self.video._percentage_of_accumulated_images = [
            #     self.video.percentage_of_accumulated_images[0]
            # ]
            # logger.info("Start accumulation parachute")
            #
            # self.accumulate()

        elif (
            "protocols1_and_2" in self.processes_to_restore
            and self.processes_to_restore["protocols1_and_2"]
        ):
            # TODO: bring restoring back to life
            raise
            # logger.info("Restoring protocol 1 and 2")
            # self.restoring_first_accumulation = True
            # # self.restore_first_accumulation()
            # self.accumulation_manager.ratio_accumulated_images =
            # self.video.percentage_of_accumulated_images[0]
            # self.video._first_frame_first_global_fragment = [
            #     self.video._first_frame_first_global_fragment[
            #         0
            #     ]
            # ]
            # self.video._percentage_of_accumulated_images = [
            #     self.video.percentage_of_accumulated_images[0]
            # ]
            # self.accumulation_step_finished = True
            #
            # self.accumulate()

        elif (
            "protocols1_and_2" not in self.processes_to_restore
            or not self.processes_to_restore["protocols1_and_2"]
        ):
            logger.info("Starting protocol cascade")
            self.protocol1()

    def init_accumulation_idCNN_params(self):
        self.accumulation_network_params = NetworkParams(
            number_of_classes=self.video.user_defined_parameters[
                "number_of_animals"
            ],
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.accumulation_folder,
            knowledge_transfer_model_file=self.video.user_defined_parameters[
                "knowledge_transfer_folder"
            ],
            saveid="",
            model_name="identification_network",
            image_size=self.video.identification_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            loss="CE",
            print_freq=-1,
            use_gpu=True,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION},
            apply_mask=False,
            dataset="supervised",
            skip_eval=False,
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            plot_flag=False,
            return_store_objects=False,
            layers_to_optimize=conf.LAYERS_TO_OPTIMISE_ACCUMULATION,
            video_path=self.video.video_path,
        )
        # Save network params
        self.accumulation_network_params.save()

    def protocol1(self):
        logger.debug("****** setting protocol1 time")
        # set timer
        self.video._protocol1_time = time.time()

        # reset list of fragments and global fragments to fragmentation
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.reset(roll_back_to="fragmentation")

        # Initialize idCNN
        logger.info("Setting learner class")
        self.learner_class = Learner_Classification
        logger.info("Creating idCNN")
        if self.video.user_defined_parameters.get(
            "knowledge_transfer_folder", False
        ):
            logger.info("Tracking with knowledge transfer")
            self.identification_model = self.learner_class.load_model(
                self.accumulation_network_params, scope="knowledge_transfer"
            )
            if not self.video.user_defined_parameters["identity_transfer"]:
                logger.info("Reinitializing fully connected layers")
                self.identification_model.apply(fc_weights_reinit)
            else:
                logger.info(
                    "Identity transfer. Not reinitializing the fully connected layers."
                )
        else:
            self.identification_model = self.learner_class.create_model(
                self.accumulation_network_params
            )
            self.identification_model.apply(weights_xavier_init)

        # Set first global fragment to start accumulation.
        # The network is passed in case of identity transfer.
        logger.info("Setting first global fragment for accumulation")
        self.video._first_frame_first_global_fragment.append(
            self.list_of_global_fragments.set_first_global_fragment_for_accumulation(
                self.video,
                identification_model=self.identification_model,
                accumulation_trial=0,
                network_params=self.accumulation_network_params,
                knowledge_transfer_info_dict=self.knowledge_transfer_info_dict,
            )
        )

        # TODO: Perform identity transfer here instead of in the previous method

        # Order global fragments by distance to the first global fragment for the accumulation
        logger.info("Setting first global fragment for accumulation")
        self.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(
            self.video, accumulation_trial=0
        )

        # Instantiate accumulation manager
        self.accumulation_manager = AccumulationManager(
            self.video,
            self.list_of_fragments,
            self.list_of_global_fragments,
            threshold_acceptable_accumulation=conf.THRESHOLD_ACCEPTABLE_ACCUMULATION,
        )

        # General counter for training epochs
        self.global_step = 0

        # Selecting the first global fragment is considered as
        # the 0 accumulation step
        self.accumulation_step_finished = True
        self.init_and_accumulate()

    def one_shot_accumulation(self):
        logger.info("Starting one_shot_accumulation")
        self.accumulation_step_finished = False
        self.accumulation_manager.ratio_accumulated_images = (
            perform_one_accumulation_step(
                self.accumulation_manager,
                self.video,
                self.identification_model,
                self.learner_class,
                network_params=self.accumulation_network_params,
            )
        )
        self.accumulation_step_finished = True

    def accumulate(self):
        logger.info("------------------------> Calling accumulate")

        if (
            self.accumulation_step_finished
            and self.accumulation_manager.new_global_fragments_for_training
        ):
            # Training and identification continues
            logger.info("--------------------> Performing accumulation")
            if (
                self.accumulation_manager.counter == 1
                and self.video.accumulation_trial == 0
            ):
                # first training finished
                # Measure time of protocol 1
                self.video._protocol1_time = (
                    time.time() - self.video.protocol1_time
                )
                self.video._has_protocol1_finished = True
                # Start timer of protocol 2
                self.video._protocol2_time = time.time()

            # Training and identification step
            self.one_shot_accumulation()
            # Re-enter the function for the next step of the accumulation
            self.accumulate()

        elif (
            not self.accumulation_manager.new_global_fragments_for_training
            and not self.video._has_protocol2_finished
            and self.accumulation_manager.ratio_accumulated_images
            > conf.THRESHOLD_EARLY_STOP_ACCUMULATION
        ):
            # Accumulation stop because protocol 1 is successful
            logger.info("--------------------> Protocol 1 successful")
            self.save_after_first_accumulation()
            if (
                "protocols1_and_2" not in self.processes_to_restore
                or not self.processes_to_restore["protocols1_and_2"]
            ):
                self.video._protocol1_time = (
                    time.time() - self.video.protocol1_time
                )

            self.identify()
            self.postprocess_impossible_jumps()

        elif (
            not self.accumulation_manager.new_global_fragments_for_training
            and not self.video.has_protocol3_pretraining_finished
        ):
            logger.info("--------------------> No more new global fragments")
            self.save_after_first_accumulation()

            if (
                self.accumulation_manager.ratio_accumulated_images
                >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            ):
                logger.info("--------------------> Protocol 2 successful")

                self.save_after_first_accumulation()
                if (
                    "protocols1_and_2" not in self.processes_to_restore
                    or not self.processes_to_restore["protocols1_and_2"]
                ):
                    self.video._protocol2_time = (
                        time.time() - self.video.protocol2_time
                    )

                self.identify()
                self.postprocess_impossible_jumps()

            elif (
                self.accumulation_manager.ratio_accumulated_images
                < conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            ):

                logger.info(
                    "--------------------> Protocol 2 failed -> Start protocol 3"
                )
                # raise ValueError('Protocol 3')
                if (
                    "protocols1_and_2" not in self.processes_to_restore
                    or not self.processes_to_restore["protocols1_and_2"]
                ):
                    self.video._protocol1_time = (
                        time.time() - self.video.protocol1_time
                    )
                    if self.video.protocol2_time != 0:
                        self.video._protocol2_time = (
                            time.time() - self.video.protocol2_time
                        )
                self.video._protocol3_pretraining_time = time.time()

                self.pretraining_counter = 0
                self.protocol3()

        elif (
            self.video.has_protocol3_pretraining_finished
            and self.video.accumulation_trial
            < conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
            and self.accumulation_manager.ratio_accumulated_images
            < conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
        ):

            logger.info(
                "--------------------> Accumulation Protocol 3 failed. Opening parachute ..."
            )
            if self.video.accumulation_trial == 0:
                self.video._protocol3_accumulation_time = time.time()
            self.video._accumulation_trial += 1
            if (
                not self.accumulation_manager.new_global_fragments_for_training
                and self.video.accumulation_trial > 1
            ):
                self.save_and_update_accumulation_parameters_in_parachute()
            self.accumulation_parachute_init(self.video.accumulation_trial)

            self.init_and_accumulate()

        elif self.video.has_protocol3_pretraining_finished and (
            self.accumulation_manager.ratio_accumulated_images
            >= conf.THRESHOLD_ACCEPTABLE_ACCUMULATION
            or self.video.accumulation_trial
            >= conf.MAXIMUM_NUMBER_OF_PARACHUTE_ACCUMULATIONS
        ):

            logger.info(
                "--------------------> Accumulation after protocol 3 has been successful"
            )
            if "protocol3_accumulation" not in self.processes_to_restore:
                self.video._protocol3_accumulation_time = (
                    time.time() - self.video.protocol3_accumulation_time
                )
            elif (
                "protocol3_accumulation" in self.processes_to_restore
                and not self.processes_to_restore["protocol3_accumulation"]
            ):
                self.video._protocol3_accumulation_time = (
                    time.time() - self.video.protocol3_accumulation_time
                )
            else:
                self.video._protocol3_accumulation_time = (
                    time.time() - self.video.protocol3_accumulation_time
                )

            self.save_after_second_accumulation()
            logger.info("Start residual indentification")
            self.identify()
            self.postprocess_impossible_jumps()

        # Whether to re-enter the function for the next accumulation step
        if self.accumulation_manager.new_global_fragments_for_training:
            self.accumulate()

    def init_and_accumulate(self):
        """
        This is called in the first step of each accumulation trial
        :param do_accumulate:
        :return:
        """
        logger.warning("------------Calling accumulation loop")
        self.video.init_accumulation_statistics_attributes()
        self.accumulation_manager.threshold_early_stop_accumulation = (
            conf.THRESHOLD_EARLY_STOP_ACCUMULATION
        )
        logger.warning("Calling accumulate from init_and_accumulate")
        self.accumulate()

    def save_after_first_accumulation(self):
        """Set flags and save data"""
        logger.info("Saving first accumulation paramters")

        if not self.restoring_first_accumulation:
            self.video._first_accumulation_finished = True
            self.video._ratio_accumulated_images = (
                self.accumulation_manager.ratio_accumulated_images
            )
            self.video._percentage_of_accumulated_images = [
                self.video.ratio_accumulated_images
            ]
            self.video._accumulation_network_params = (
                self.accumulation_network_params
            )
            self.video.save()
            self.list_of_fragments.save(self.video.fragments_path)
            self.list_of_global_fragments.save(
                self.video.global_fragments_path,
                self.list_of_fragments.fragments,
            )
            self.list_of_fragments.save_light_list(
                self.video._accumulation_folder
            )

    """ pretraining """

    def protocol3(self):

        self.init_pretraining_variables()

        logger.info(
            "Starting pretraining. Checkpoints will be stored in %s"
            % self.video.pretraining_folder
        )

        if self.video.user_defined_parameters.get(
            "knowledge_transfer_folder", False
        ):
            logger.info(
                "Performing knowledge transfer from %s"
                % self.video.user_defined_parameters[
                    "knowledge_transfer_folder"
                ]
            )
            self.pretrain_network_params.knowledge_transfer_model_file = (
                self.video.user_defined_parameters["knowledge_transfer_folder"]
            )

        logger.info("Start pretraining")
        self.pretraining_step_finished = True
        self.pretraining_loop()

    def init_pretraining_variables(self):

        self.init_pretraining_net()
        self.pretraining_global_step = 0
        self.ratio_of_pretrained_images = 0

        # Initialize network
        self.learner_class = Learner_Classification
        logger.info("Creating model")
        if self.video.user_defined_parameters.get(
            "knowledge_transfer_folder", False
        ):
            self.identification_model = self.learner_class.load_model(
                self.pretrain_network_params, scope="knowledge_transfer"
            )
            self.identification_model.apply(fc_weights_reinit)
        else:
            self.identification_model = self.learner_class.create_model(
                self.pretrain_network_params
            )
            self.identification_model.apply(weights_xavier_init)

    def init_pretraining_net(self):
        delete = (
            not self.processes_to_restore["protocol3_pretraining"]
            if "protocol3_pretraining" in self.processes_to_restore.keys()
            else True
        )
        self.video.create_pretraining_folder(delete=delete)

        self.pretrain_network_params = NetworkParams(
            number_of_classes=self.video.user_defined_parameters[
                "number_of_animals"
            ],
            architecture=conf.IDCNN_NETWORK_NAME,
            save_folder=self.video.pretraining_folder,
            saveid="",
            model_name="identification_network",
            image_size=self.video.identification_image_size,
            scopes_layers_to_optimize=conf.LAYERS_TO_OPTIMISE_PRETRAINING,
            loss="CE",
            print_freq=-1,
            use_gpu=True,
            optimizer="SGD",
            schedule=[30, 60],
            optim_args={"lr": conf.LEARNING_RATE_IDCNN_ACCUMULATION},
            apply_mask=False,
            dataset="supervised",
            skip_eval=False,
            epochs=conf.MAXIMUM_NUMBER_OF_EPOCHS_IDCNN,
            plot_flag=False,
            return_store_objects=False,
            layers_to_optimize=conf.LAYERS_TO_OPTIMISE_ACCUMULATION,
            video_path=self.video.video_path,
        )
        self.video._pretraining_network_params = self.pretrain_network_params

    def pretraining_loop(self):
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.order_by_distance_travelled()
        self.one_shot_pretraining()
        self.continue_pretraining()

    def one_shot_pretraining(self):
        self.pretraining_step_finished = False
        self.pretraining_global_fragment = (
            self.list_of_global_fragments.global_fragments[
                self.pretraining_counter
            ]
        )
        (
            self.identification_model,
            self.ratio_of_pretrained_images,
            pretraining_global_step,
            self.list_of_fragments,
            self.pretrained_model_path,
        ) = pre_train_global_fragment(
            self.video,
            self.identification_model,
            self.learner_class,
            self.pretrain_network_params,
            self.pretraining_global_fragment,
            self.list_of_fragments,
            self.pretraining_global_step,
        )
        self.pretraining_counter += 1
        self.pretraining_step_finished = True

    def continue_pretraining(self, clock_unschedule=None):
        if (
            self.pretraining_step_finished
            and self.ratio_of_pretrained_images
            < conf.MAX_RATIO_OF_PRETRAINED_IMAGES
        ):
            self.one_shot_pretraining()

            if clock_unschedule is None:
                self.continue_pretraining()

        elif (
            self.ratio_of_pretrained_images
            > conf.MAX_RATIO_OF_PRETRAINED_IMAGES
        ):
            self.video._has_protocol3_pretraining_finished = True

            logger.warning("Calling accumulate from continue_pretraining")
            logger.debug("****** saving protocol3 pretraining time")
            self.video._protocol3_pretraining_time = (
                time.time() - self.video.protocol3_pretraining_time
            )
            self.accumulate()

    """ parachute """

    def accumulation_parachute_init(self, iteration_number):
        logger.debug("------------------------> accumulation_parachute_init")
        logger.info("Starting accumulation %i" % iteration_number)

        delete = (
            not self.processes_to_restore["protocol3_accumulation"]
            if "protocol3_accumulation" in self.processes_to_restore.keys()
            else True
        )

        self.video.create_accumulation_folder(
            iteration_number=iteration_number, delete=delete
        )
        self.video._accumulation_trial = iteration_number
        self.list_of_fragments.reset(roll_back_to="fragmentation")
        self.list_of_global_fragments.reset(roll_back_to="fragmentation")

        # Initialize network
        if self.video.user_defined_parameters["identity_transfer"]:
            logger.info("Load model for identity transfer")
            self.identification_model = self.learner_class.load_model(
                self.accumulation_network_params
            )
        else:
            self.identification_model = None

        # Choose first global fragment
        self.video._first_frame_first_global_fragment.append(
            self.list_of_global_fragments.set_first_global_fragment_for_accumulation(
                self.video,
                identification_model=self.identification_model,
                accumulation_trial=iteration_number - 1,
                network_params=self.accumulation_network_params,
                knowledge_transfer_info_dict=self.knowledge_transfer_info_dict,
            )
        )

        # Sort global fragments by distance
        self.list_of_global_fragments.order_by_distance_to_the_first_global_fragment_for_accumulation(
            self.video, accumulation_trial=iteration_number - 1
        )
        logger.warning(
            "first_frame_first_global_fragment "
            + str(self.video.first_frame_first_global_fragment)
        )
        logger.info(
            "We will restore the network from a previous pretraining: %s"
            % self.video.pretraining_folder
        )

        # Set saving folders
        self.accumulation_network_params.save_folder = (
            self.video.accumulation_folder
        )

        # Set restoring model_file
        self.accumulation_network_params.restore_folder = (
            self.video.pretraining_folder
        )

        # TODO: allow to train only the fully connected layers
        self.accumulation_network_params.scopes_layers_to_optimize = [
            "fully-connected1",
            "fully_connected_pre_softmax",
        ]
        logger.info("Initialising accumulation network")

        # Load pretrained network
        self.identification_model = self.learner_class.load_model(
            self.accumulation_network_params
        )

        # Re-initialize fully-connected layers
        self.identification_model.apply(fc_weights_reinit)

        # Instantiate accumualtion manager
        logger.info("Initialising accumulation manager")
        self.accumulation_manager = AccumulationManager(
            self.video,
            self.list_of_fragments,
            self.list_of_global_fragments,
            threshold_acceptable_accumulation=conf.THRESHOLD_ACCEPTABLE_ACCUMULATION,
        )

        logger.info("Start accumulation")
        self.global_step = 0

    def save_and_update_accumulation_parameters_in_parachute(self):
        logger.warning(
            "self.accumulation_manager.ratio_accumulated_images %.4f"
            % self.accumulation_manager.ratio_accumulated_images
        )
        self.video._ratio_accumulated_images = (
            self.accumulation_manager.ratio_accumulated_images
        )
        self.video._percentage_of_accumulated_images.append(
            self.video.ratio_accumulated_images
        )
        self.list_of_fragments.save_light_list(self.video._accumulation_folder)

    def save_after_second_accumulation(self):
        logger.info("Saving second accumulation parameters")
        # Save accumulation parameters
        self.save_and_update_accumulation_parameters_in_parachute()

        # Choose best accumulation
        self.video._accumulation_trial = np.argmax(
            self.video.percentage_of_accumulated_images
        )

        # Update ratio of accumulated images and  accumulation folder
        self.video._ratio_accumulated_images = (
            self.video.percentage_of_accumulated_images[
                self.video._accumulation_trial
            ]
        )
        accumulation_folder_name = "accumulation_" + str(
            self.video._accumulation_trial
        )
        self.video._accumulation_folder = os.path.join(
            self.video.session_folder, accumulation_folder_name
        )

        # Load light list of fragments with identities of the best accumulation
        self.list_of_fragments.load_light_list(self.video._accumulation_folder)

        # Save objects
        self.video._second_accumulation_finished = True
        logger.info("Saving global fragments")
        self.list_of_fragments.save(self.video.fragments_path)
        self.list_of_global_fragments.save(
            self.video.global_fragments_path,
            self.list_of_fragments.fragments,
        )

        # set restoring folder
        logger.info("Restoring networks to best second accumulation")
        self.accumulation_network_params.restore_folder = (
            self.video._accumulation_folder
        )

        # TODO: allow to train only the fully connected layers
        self.accumulation_network_params.scopes_layers_to_optimize = [
            "fully-connected1",
            "fully_connected_pre_softmax",
        ]
        logger.info("Initialising accumulation network")

        # Load pretrained network
        self.identification_model = self.learner_class.load_model(
            self.accumulation_network_params
        )

        # # Re-initialize fully-connected layers
        # self.identification_model.apply(fc_weights_reinit)

        # Send model and criterion to GPU
        if self.accumulation_network_params.use_gpu:
            logger.info("Sending model and criterion to GPU")
            torch.cuda.set_device(0)
            cudnn.benchmark = True  # make it train faster
            self.identification_model = self.identification_model.cuda()

        self.video._accumulation_network_params = (
            self.accumulation_network_params
        )
        self.video.save()

    """ Residual identification """

    def identify(self):
        self.video._identify_time = time.time()
        logger.warning("In identify")
        self.list_of_fragments.reset(roll_back_to="accumulation")
        logger.warning("Assigning remaining fragments")
        assign_remaining_fragments(
            self.list_of_fragments,
            self.video,
            self.identification_model,
            self.accumulation_network_params,
        )
        self.video.save()

    """ Post processing """

    def postprocess_impossible_jumps(self, call_update_list_of_blobs=True):
        self.video.velocity_threshold = compute_model_velocity(
            self.list_of_fragments.fragments,
            self.video.user_defined_parameters["number_of_animals"],
            percentile=conf.VEL_PERCENTILE,
        )
        correct_impossible_velocity_jumps(self.video, self.list_of_fragments)
        self.list_of_fragments.save(self.video.fragments_path)
        self.video.save()

        if call_update_list_of_blobs:
            self.update_list_of_blobs()

    def update_list_of_blobs(self, create_trajectories=None):

        if create_trajectories is None:
            create_trajectories = self.create_trajectories

        self.video.individual_fragments_stats = (
            self.list_of_fragments.get_stats(self.list_of_global_fragments)
        )
        self.video.compute_estimated_accuracy(self.list_of_fragments.fragments)
        self.list_of_fragments.save_light_list(self.video._accumulation_folder)
        self.video.save()
        self.list_of_blobs.update_from_list_of_fragments(
            self.list_of_fragments.fragments,
            self.video.fragment_identifier_to_index,
        )
        # if False:
        #     self.list_of_blobs.compute_nose_and_head_coordinates()
        self.list_of_blobs.save(self.video.blobs_path)
        self.video._identify_time = time.time() - self.video.identify_time
        create_trajectories()

    def create_trajectories(
        self,
        trajectories_popup_dismiss=None,
        interpolate_crossings=None,
        update_and_show_happy_ending_popup=None,
    ):

        if interpolate_crossings is None:
            interpolate_crossings = self.interpolate_crossings

        self.video._create_trajectories_time = time.time()
        if (
            "post_processing" not in self.processes_to_restore
            or not self.processes_to_restore["post_processing"]
        ):
            if not self.video.user_defined_parameters[
                "track_wo_identification"
            ]:
                self.video.create_trajectories_folder()
                trajectories_file = os.path.join(
                    self.video.trajectories_folder,
                    "trajectories.npy",
                )
                trajectories = produce_output_dict(
                    self.list_of_blobs.blobs_in_video,
                    self.video,
                )
            else:
                self.video.create_trajectories_wo_identification_folder()
                trajectories_file = os.path.join(
                    self.video.trajectories_wo_identification_folder,
                    "trajectories_wo_identification.npy",
                )
                trajectories = produce_output_dict(
                    self.list_of_blobs.blobs_in_video,
                    self.video,
                )
            logger.info("Saving trajectories")
            np.save(trajectories_file, trajectories)
            if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
                logger.info("Saving trajectories in csv format...")
                convert_trajectories_file_to_csv_and_json(trajectories_file)

        self.video._has_trajectories = True

        # Call GUI function
        if trajectories_popup_dismiss:
            trajectories_popup_dismiss()

        if (
            not self.video.user_defined_parameters["track_wo_identification"]
            and self.video.user_defined_parameters["number_of_animals"] != 1
            and self.list_of_global_fragments.number_of_global_fragments != 1
        ):
            # Call GUI function
            interpolate_crossings()
        else:
            self.video._estimated_accuracy = 1.0
            self.video._has_crossings_solved = False
            self.video._has_trajectories_wo_gaps = False
            self.list_of_blobs.save(self.video.blobs_path)
            # Call GUI function
            if update_and_show_happy_ending_popup:
                update_and_show_happy_ending_popup()
        self.video.save()

    def interpolate_crossings(self, interpolate_crossings_popups_actions=None):

        if interpolate_crossings_popups_actions is None:
            interpolate_crossings_popups_actions = (
                self.create_trajectories_wo_gaps
            )

        self.list_of_blobs_no_gaps = copy.deepcopy(self.list_of_blobs)
        self.video._has_crossings_solved = False
        self.list_of_blobs_no_gaps = close_trajectories_gaps(
            self.video,
            self.list_of_blobs_no_gaps,
            self.list_of_fragments,
        )
        self.video.blobs_no_gaps_path = os.path.join(
            os.path.split(self.video.blobs_path)[0],
            "blobs_collection_no_gaps.npy",
        )
        self.list_of_blobs_no_gaps.save(
            path_to_save=self.video.blobs_no_gaps_path,
        )
        self.video._has_crossings_solved = True
        self.video.save()

        interpolate_crossings_popups_actions()

    def create_trajectories_wo_gaps(self):
        self.video.create_trajectories_wo_gaps_folder()
        logger.info(
            "Generating trajectories. The trajectories files are stored in %s"
            % self.video.trajectories_wo_gaps_folder
        )
        trajectories_wo_gaps_file = os.path.join(
            self.video.trajectories_wo_gaps_folder,
            "trajectories_wo_gaps.npy",
        )
        trajectories_wo_gaps = produce_output_dict(
            self.list_of_blobs_no_gaps.blobs_in_video,
            self.video,
        )
        np.save(trajectories_wo_gaps_file, trajectories_wo_gaps)
        if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
            logger.info("Saving trajectories in csv format...")
            convert_trajectories_file_to_csv_and_json(
                trajectories_wo_gaps_file
            )
        self.video._has_trajectories_wo_gaps = True
        logger.info("Saving trajectories")
        self.list_of_blobs = assign_zeros_with_interpolation_identities(
            self.list_of_blobs,
            self.list_of_blobs_no_gaps,
        )
        trajectories_file = os.path.join(
            self.video.trajectories_folder, "trajectories.npy"
        )
        trajectories = produce_output_dict(
            self.list_of_blobs.blobs_in_video,
            self.video,
        )
        np.save(trajectories_file, trajectories)
        if conf.CONVERT_TRAJECTORIES_DICT_TO_CSV_AND_JSON:
            logger.info("Saving trajectories in csv format...")
            convert_trajectories_file_to_csv_and_json(trajectories_file)
        self.video.save()
        self.video._create_trajectories_time = (
            time.time() - self.video.create_trajectories_time
        )

    def update_and_show_happy_ending_popup(self):
        if not hasattr(self.video, "estimated_accuracy"):
            self.video.compute_estimated_accuracy(
                self.list_of_fragments.fragments
            )
        self.video.save()
