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

from tqdm import tqdm
import matplotlib.pyplot as plt
from confapp import conf

from idtrackerai.network.identification_model.get_data import split_data_train_and_validation
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.network.identification_model.epoch_runner import EpochRunner
from idtrackerai.network.identification_model.stop_training_criteria import Stop_Training
from idtrackerai.network.identification_model.store_accuracy_and_loss import Store_Accuracy_and_Loss

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.pre_trainer")

def pre_train_global_fragment(net,
                                pretraining_global_fragment,
                                list_of_fragments,
                                global_epoch,
                                check_for_loss_plateau,
                                store_accuracy_and_error,
                                save_summaries, store_training_accuracy_and_loss_data,
                                store_validation_accuracy_and_loss_data,
                                print_flag = False,
                                plot_flag = False,
                                batch_size = None,
                                canvas_from_GUI = None):
    """Performs pretraining on a single global fragments

    Parameters
    ----------
    net : <ConvNetwork obejct>
        an instance of the class :class:`~idCNN.ConvNetwork`
    pretraining_global_fragment : <GlobalFragment object>
        an instance of the class :class:`~globalfragment.GlobalFragment`
    list_of_fragments : <ListOfFragments object>
        an instance of the class :class:`~list_of_fragments.ListOfFragments`
    global_epoch : int
        global counter of the training epoch in pretraining
    check_for_loss_plateau : bool
        if True the stopping criteria (see :mod:`~stop_training_criteria`) will
        automatically stop the training in case the loss functin computed for
        the validation set of images reaches a plateau
    store_accuracy_and_error : bool
        if True the values of the loss function, accuracy and individual
        accuracy will be stored
    save_summaries : bool
        if True tensorflow summaries will be generated and stored to allow
        tensorboard visualisation of both loss and activity histograms
    store_training_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`
    store_validation_accuracy_and_loss_data : <Store_Accuracy_and_Loss object>
        an instance of the class :class:`~Store_Accuracy_and_Loss`
    print_flag : bool
        if True additional information are printed in the terminal
    plot_flag : bool
        if True training and validation loss, accuracy and individual accuracy
        are plot in a graph at the end of the training session
    batch_size : int
        size of the batch of images used for training
    canvas_from_GUI : matplotlib figure canvas
        canvas of the matplotlib figure initialised in
        :class:`~tracker.Tracker` used to update the figure in the GUI
        visualisation of pretraining

    Returns
    -------
    <ConvNetwork object>
        network with updated parameters after training
    float
        ration of images used for pretraining over the total number of
        available images
    int
        global epoch counter updated after the training session
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the training set of labelled
        images
    <Store_Accuracy_and_Loss object>
        updated with the values collected on the validation set of labelled
        images
    <ListOfFragments objects>
        list of instances of the class :class:`~fragment.Fragment`
    """
    # Get images and labels from the current global fragment
    images, labels = pretraining_global_fragment.get_images_and_labels(list_of_fragments.identification_images_file_path)
    # Instantiate data_set
    training_dataset, validation_dataset = split_data_train_and_validation(net.params.number_of_animals,
                                                                            images, labels)
    training_dataset.convert_labels_to_one_hot()
    validation_dataset.convert_labels_to_one_hot()
    # Reinitialize softmax and fully connected
    # (the fully connected layer and the softmax are initialized since the labels
    # of the images for each pretraining global fragments are different)
    net.reinitialize_softmax_and_fully_connected()
    # Train network
    #compute weights to be fed to the loss function (weighted cross entropy)
    net.compute_loss_weights(training_dataset.labels)
    #instantiate epochs runners for train and validation
    trainer = EpochRunner(training_dataset,
                        starting_epoch = global_epoch,
                        print_flag = print_flag,
                        batch_size = batch_size)
    validator = EpochRunner(validation_dataset,
                        starting_epoch = global_epoch,
                        print_flag = print_flag,
                        batch_size = batch_size)
    #set criteria to stop the training
    stop_training = Stop_Training(net.params.number_of_animals,
                                check_for_loss_plateau = check_for_loss_plateau)

    while not stop_training(store_training_accuracy_and_loss_data,
                            store_validation_accuracy_and_loss_data,
                            trainer._epochs_completed):
        feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, net.train)
        feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, net.validate)
        net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed))
        if save_summaries:
            net.write_summaries(trainer.starting_epoch + trainer._epochs_completed,feed_dict_train, feed_dict_val)
        trainer._epochs_completed += 1
        validator._epochs_completed += 1
    pretraining_global_fragment.update_individual_fragments_attribute('_used_for_pretraining', True)
    if plot_flag and canvas_from_GUI is None:
        store_training_accuracy_and_loss_data.plot_global_fragments(ax_arr, video, list_of_fragments.fragments, black = False)
        ax_arr[2].cla() # clear bars
        store_training_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'r')
        store_validation_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'b')
        epoch_index_to_plot += trainer._epochs_completed
    if store_accuracy_and_error:
        store_training_accuracy_and_loss_data.save(trainer._epochs_completed)
        store_validation_accuracy_and_loss_data.save(trainer._epochs_completed)
    global_epoch += trainer._epochs_completed
    net.save()
    if plot_flag:
        fig.savefig(os.path.join(net.params.save_folder,'pretraining_gf%i.pdf'%i))
    ratio_of_pretrained_images = list_of_fragments.compute_ratio_of_images_used_for_pretraining()
    logger.debug("limit ratio of images to be used during pretraining: %.4f (if higher than %.2f we stop)" %(ratio_of_pretrained_images, conf.MAX_RATIO_OF_PRETRAINED_IMAGES))
    return net, ratio_of_pretrained_images, global_epoch, store_training_accuracy_and_loss_data, store_validation_accuracy_and_loss_data, list_of_fragments

def pre_train(video, list_of_fragments, list_of_global_fragments,
                params, store_accuracy_and_error,
                check_for_loss_plateau, save_summaries,
                print_flag, plot_flag):
    """Performs pretraining by iterating on the list of global fragments
    sorted by distance travelled, until the threshold
    :const:`conf.MAX_RATIO_OF_PRETRAINED_IMAGES` is reached

    Parameters
    ----------
    video : <Video object>
        an instance of the class :class:`~video.Video`
    list_of_fragments : <ListOfFragments object>
        an instance of the class :class:`~list_of_fragments.ListOfFragments`
    list_of_global_fragments : <ListOfGlobalFragments object>
        an instance of the class
        :class:`~list_of_global_fragments.ListOfGlobalFragments`
    params : <NetworkParams object>
        an instance of the class :class:`~network_params.NetworkParams`
    store_accuracy_and_error : bool
        if True the values of the loss function, accuracy and individual
        accuracy will be stored
    check_for_loss_plateau : bool
        if True the stopping criteria (see :mod:`~stop_training_criteria`) will
        automatically stop the training in case the loss functin computed for
        the validation set of images reaches a plateau
    save_summaries : bool
        if True tensorflow summaries will be generated and stored to allow
        tensorboard visualisation of both loss and activity histograms
    print_flag : bool
        if True additional information are printed in the terminal
    plot_flag : bool
        if True training and validation loss, accuracy and individual accuracy
        are plot in a graph at the end of the training session


    Returns
    -------
    <ConvNetwork object>
        an instance of the class :class:`~id_CNN.ConvNetwork`

    """
    #initialize global epoch counter that takes into account all the steps in the pretraining
    global_epoch = 0
    number_of_images_used_during_pretraining = 0
    #initialize network
    net = ConvNetwork(params)
    if video.tracking_with_knowledge_transfer:
        net.restore()
    #instantiate objects to store loss and accuracy values for training and validation
    #(the loss and accuracy of the validation are saved to allow the automatic stopping of the training)
    store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net,
                                                                    name = 'training',
                                                                    scope = 'pretraining')
    store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net,
                                                                    name = 'validation',
                                                                    scope = 'pretraining')
    if plot_flag:
        plt.ion()
        fig, ax_arr = plt.subplots(4)
        fig.canvas.set_window_title('Pretraining')
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        epoch_index_to_plot = 0

    for i, pretraining_global_fragment in enumerate(tqdm(list_of_global_fragments.global_fragments, desc = '\nPretraining network')):
        net, ratio_of_pretrained_images, global_epoch, _, _, _ = pre_train_global_fragment(net,
                                                                            pretraining_global_fragment,
                                                                            list_of_fragments,
                                                                            global_epoch,
                                                                            check_for_loss_plateau,
                                                                            store_accuracy_and_error,
                                                                            save_summaries, store_training_accuracy_and_loss_data,
                                                                            store_validation_accuracy_and_loss_data,
                                                                            print_flag = print_flag,
                                                                            plot_flag = plot_flag,
                                                                            batch_size = conf.BATCH_SIZE_IDCNN)
        if ratio_of_pretrained_images > conf.MAX_RATIO_OF_PRETRAINED_IMAGES:
            logger.info("pre-training ended: The network has been pre-trained on more than %.4f of the images in global fragment" %conf.MAX_RATIO_OF_PRETRAINED_IMAGES)
            break

    return net

# def pre_trainer(old_video, video, list_of_fragments, list_of_global_fragments, pretrain_network_params):
#     """Initialises and starts the pretraining (3rd fingerprint protocol)
#
#     Parameters
#     ----------
#     old_video : <Video object>
#         an instance of the class :class:`~video.Video`
#     video :<Video object>
#         an instance of the class :class:`~video.Video`
#     list_of_fragments : <ListOfFragments object>
#         an instance of the class :class:`~list_of_fragments.ListOfFragments`
#     list_of_global_fragments : <ListOfGlobalFragments object>
#         an instance of the class
#         :class:`~list_of_global_fragments.ListOfGlobalFragments`
#     pretrain_network_params :  <NetworkParams object>
#         an instance of the class :class:`~network_params.NetworkParams`
#     """
#     #Reset used_for_training and acceptable_for_training flags
#     if old_video and old_video.first_accumulation_finished == True:
#         list_of_global_fragments.reset(roll_back_to = 'fragmentation')
#         list_of_fragments.reset(roll_back_to = 'fragmentation')
#     logger.info("Starting pretraining. Checkpoints will be stored in %s" %video.pretraining_folder)
#     if video.tracking_with_knowledge_transfer:
#         logger.info("Performing knowledge transfer from %s" %video.knowledge_transfer_model_folder)
#         pretrain_network_params.knowledge_transfer_folder = video.knowledge_transfer_model_folder
#     #start pretraining
#     logger.info("Start pretraining")
#     net = pre_train(video, list_of_fragments,
#                     list_of_global_fragments,
#                     pretrain_network_params,
#                     store_accuracy_and_error = True,
#                     check_for_loss_plateau = True,
#                     save_summaries = False,
#                     print_flag = False,
#                     plot_flag = False)
