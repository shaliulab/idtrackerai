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
from globalfragment import get_images_and_labels_from_global_fragment,\
                        give_me_pre_training_global_fragments,\
                        get_number_of_images_in_global_fragments_list,\
                        order_global_fragments_by_distance_travelled,\
                        give_me_number_of_unique_images_in_global_fragments
from epoch_runner import EpochRunner
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

MAX_RATIO_OF_PRETRAINED_IMAGES = .95

logger = logging.getLogger("__main__.pre_trainer")

def pre_train(video, blobs_in_video, number_of_images_in_global_fragments, pretraining_global_fragments, params, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag, plot_flag):
    #initialize global epoch counter that takes into account all the steps in the pretraining
    global_epoch = 0
    number_of_images_used_during_pretraining = 0
    #initialize network
    net = ConvNetwork(params)
    if video.tracking_with_knowledge_transfer:
        net.restore()
    #instantiate objects to store loss and accuracy values for training and validation
    #(the loss and accuracy of the validation are saved to allow the automatic stopping of the training)
    store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training')
    store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation')
    #open figure for plotting
    if plot_flag:
        plt.ion()
        fig, ax_arr = plt.subplots(4)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        epoch_index_to_plot = 0
    #start loop for pre training in the global fragments selected
    for i, pretraining_global_fragment in enumerate(tqdm(pretraining_global_fragments, desc = '\nPretraining network')):
        # Get images and labels from the current global fragment
        images, labels, _, _ = get_images_and_labels_from_global_fragment(pretraining_global_fragment)
        # Instantiate data_set
        training_dataset, validation_dataset = split_data_train_and_validation(video.preprocessing_type, params.number_of_animals,images,labels)
        # Standarize images
        # training_dataset.standarize_images()
        # validation_dataset.standarize_images()
        # Crop images from 36x36 to 32x32 without performing data augmentation
        training_dataset.crop_images(image_size = video.portrait_size[0])
        validation_dataset.crop_images(image_size = video.portrait_size[0])
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
        if plot_flag:
            pretrained_global_fragments = pretraining_global_fragments[:i + 1]
            store_training_accuracy_and_loss_data.plot_global_fragments(ax_arr, video, blobs_in_video, pretrained_global_fragments, black = False)
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
        number_of_images_used_during_pretraining = give_me_number_of_unique_images_in_global_fragments([pretraining_global_fragment])
        ratio_pretrained_images = number_of_images_used_during_pretraining / number_of_images_in_global_fragments
        logger.info("total number of images in global fragments:  %i" %number_of_images_in_global_fragments)
        logger.info("number of images used during pretraining %i"  %number_of_images_used_during_pretraining)
        logger.debug("limit ratio of images to be used during pretraining: %.2f (if higher than %.2f we stop)" %(ratio_pretrained_images, MAX_RATIO_OF_PRETRAINED_IMAGES))
        if ratio_pretrained_images > MAX_RATIO_OF_PRETRAINED_IMAGES:
            logger.info("pre-training ended: The network has been pre-trained on more than %.2f of the images in global fragment" %MAX_RATIO_OF_PRETRAINED_IMAGES)
            break
    return net

def pre_trainer(old_video, video, blobs, global_fragments, pretrain_network_params):
    number_of_images_in_global_fragments = video.number_of_unique_images_in_global_fragments
    #Reset used_for_training and acceptable_for_training flags
    if old_video and old_video._first_accumulation_finished == True:
        for global_fragment in global_fragments:
            global_fragment.reset_accumulation_params()

    pretraining_global_fragments = order_global_fragments_by_distance_travelled(global_fragments)
    number_of_global_fragments = len(pretraining_global_fragments)
    logger.info("pretraining with %i global fragments" %number_of_global_fragments)
    logger.info("Starting pretraining. Checkpoints will be stored in %s" %video._pretraining_folder)
    if video.tracking_with_knowledge_transfer:
        logger.info("Performing knowledge transfer from %s" %video.knowledge_transfer_model_folder)
        pretrain_network_params.restore_folder = video.knowledge_transfer_model_folder
    #start pretraining
    logger.info("Start pretraining")
    net = pre_train(video, blobs,
                    number_of_images_in_global_fragments,
                    pretraining_global_fragments,
                    pretrain_network_params,
                    store_accuracy_and_error = False,
                    check_for_loss_plateau = True,
                    save_summaries = False,
                    print_flag = False,
                    plot_flag = True)
