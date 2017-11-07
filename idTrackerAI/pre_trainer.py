from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')
sys.path.append('./network/identification_model')

import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

from network_params import NetworkParams
from get_data import DataSet, split_data_train_and_validation
from id_CNN import ConvNetwork
from list_of_global_fragments import ListOfGlobalFragments
from epoch_runner import EpochRunner
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

MAX_RATIO_OF_PRETRAINED_IMAGES = .95

logger = logging.getLogger("__main__.pre_trainer")

def pre_train(video, list_of_fragments,
                number_of_images_in_global_fragments, list_of_global_fragments,
                params, store_accuracy_and_error,
                check_for_loss_plateau, save_summaries,
                print_flag, plot_flag):
    #initialize global epoch counter that takes into account all the steps in the pretraining
    global_epoch = 0
    number_of_images_used_during_pretraining = 0
    #initialize network
    net = ConvNetwork(params)
    if video.tracking_with_knowledge_transfer:
        net.restore()
    #instantiate objects to store loss and accuracy values for training and validation
    #(the loss and accuracy of the validation are saved to allow the automatic stopping of the training)
    store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training', scope = 'pretraining')
    store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation', scope = 'pretraining')
    #open figure for plotting
    if plot_flag:
        plt.ion()
        fig, ax_arr = plt.subplots(4)
        fig.canvas.set_window_title('Pretraining')
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        epoch_index_to_plot = 0
    #start loop for pre training in the global fragments selected
    for i, pretraining_global_fragment in enumerate(tqdm(list_of_global_fragments.global_fragments, desc = '\nPretraining network')):
        # Get images and labels from the current global fragment
        images, labels = pretraining_global_fragment.get_images_and_labels()
        print("---------------", len(images), images[0].shape)
        print("---------------", len(labels), labels[0])
        # Instantiate data_set
        training_dataset, validation_dataset = split_data_train_and_validation(params.number_of_animals,
                                                                                images, labels)
        # Standarize images
        # training_dataset.crop_images(image_size = video.identification_image_size[0])
        # validation_dataset.crop_images(image_size = video.identification_image_size[0])
        # Convert labels to one hot vectors
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
                            print_flag = print_flag)
        validator = EpochRunner(validation_dataset,
                            starting_epoch = global_epoch,
                            print_flag = print_flag)
        #set criteria to stop the training
        stop_training = Stop_Training(params.number_of_animals,
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
        if plot_flag:
            store_training_accuracy_and_loss_data.plot_global_fragments(ax_arr, video, list_of_fragments.fragments, black = False)
            ax_arr[2].cla() # clear bars
            store_training_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'r')
            store_validation_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'b')
            epoch_index_to_plot += trainer._epochs_completed
        if store_accuracy_and_error:
            store_training_accuracy_and_loss_data.save()
            store_validation_accuracy_and_loss_data.save()
        global_epoch += trainer._epochs_completed
        net.save()
        if plot_flag:
            fig.savefig(os.path.join(net.params.save_folder,'pretraining_gf%i.pdf'%i))
        ratio_of_pretrained_images = list_of_fragments.compute_ratio_of_images_used_for_pretraining()
        logger.debug("limit ratio of images to be used during pretraining: %.2f (if higher than %.2f we stop)" %(ratio_of_pretrained_images, MAX_RATIO_OF_PRETRAINED_IMAGES))
        if ratio_of_pretrained_images > MAX_RATIO_OF_PRETRAINED_IMAGES:
            logger.info("pre-training ended: The network has been pre-trained on more than %.2f of the images in global fragment" %MAX_RATIO_OF_PRETRAINED_IMAGES)
            break
    return net


def pre_trainer(old_video, video, list_of_fragments, list_of_global_fragments, pretrain_network_params):
    number_of_images_in_global_fragments = video.number_of_unique_images_in_global_fragments
    #Reset used_for_training and acceptable_for_training flags
    if old_video and old_video.first_accumulation_finished == True:
        list_of_global_fragments.reset(roll_back_to = 'fragmentation')
        list_of_fragments.reset(roll_back_to = 'fragmentation')

    logger.info("Starting pretraining. Checkpoints will be stored in %s" %video.pretraining_folder)
    if video.tracking_with_knowledge_transfer:
        logger.info("Performing knowledge transfer from %s" %video.knowledge_transfer_model_folder)
        pretrain_network_params.restore_folder = video.knowledge_transfer_model_folder
    #start pretraining
    logger.info("Start pretraining")
    net = pre_train(video, list_of_fragments,
                    number_of_images_in_global_fragments,
                    list_of_global_fragments,
                    pretrain_network_params,
                    store_accuracy_and_error = True,
                    check_for_loss_plateau = True,
                    save_summaries = False,
                    print_flag = False,
                    plot_flag = False)
