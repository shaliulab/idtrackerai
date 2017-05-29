from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from network_params import NetworkParams
from get_data import DataSet, split_data_train_and_validation
from id_CNN import ConvNetwork
from globalfragment import get_images_and_labels_from_global_fragment, give_me_pre_training_global_fragments
from epoch_runner import EpochRunner
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

def pre_train(video, blobs_in_video, pretraining_global_fragments, params, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag, plot_flag):
    #initialize global epoch counter that takes into account all the steps in the pretraining
    global_epoch = 0
    #initialize network
    net = ConvNetwork(params)
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
        training_dataset, validation_dataset = split_data_train_and_validation(params.number_of_animals,images,labels)
        # Standarize images
        training_dataset.standarize_images()
        validation_dataset.standarize_images()
        # Crop images from 36x36 to 32x32 without performing data augmentation
        # print("\ntraining images shape, ", training_dataset.images.shape)
        training_dataset.crop_images(image_size = 32)
        # print("validation images shape, ", validation_dataset.images.shape)
        validation_dataset.crop_images(image_size = 32)
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
        #enter epochs loop
        while not stop_training(store_training_accuracy_and_loss_data,
                                store_validation_accuracy_and_loss_data,
                                trainer._epochs_completed):
            # --- Training
            feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, net.train)
            # --- Validation
            feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, net.validate)
            # update global step
            net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed))
            # write summaries if asked
            if save_summaries:
                net.write_summaries(trainer.starting_epoch + trainer._epochs_completed,feed_dict_train, feed_dict_val)
            # Update counter
            trainer._epochs_completed += 1
            validator._epochs_completed += 1


        # plot if asked
        if plot_flag:
            pretrained_global_fragments = pretraining_global_fragments[:i + 1]
            store_training_accuracy_and_loss_data.plot_global_fragments(ax_arr, video, blobs_in_video, pretrained_global_fragments)
            ax_arr[2].cla() # clear bars
            store_training_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'r')
            store_validation_accuracy_and_loss_data.plot(ax_arr, epoch_index_to_plot,'b')
            epoch_index_to_plot += trainer._epochs_completed
        # store training and validation losses and accuracies
        if store_accuracy_and_error:
            store_training_accuracy_and_loss_data.save()
            store_validation_accuracy_and_loss_data.save()
        # Update global_epoch counter
        global_epoch += trainer._epochs_completed
        # Save network model
        net.save()
        if plot_flag:
            fig.savefig(os.path.join(net.params.save_folder,'pretraining_gf%i.pdf'%i))
    return net
