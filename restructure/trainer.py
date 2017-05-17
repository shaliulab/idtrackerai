from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import matplotlib.pyplot as plt
import tensorflow as tf
from network_params import NetworkParams
from get_data import DataSet, split_data_train_and_validation
from id_CNN import ConvNetwork
from epoch_runner import EpochRunner
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

def train(net, images, labels, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag, plot_flag, global_step = 0, first_accumulation_flag = False):
    # Save accuracy and error during training and validation
    # The loss and accuracy of the validation are saved to allow the automatic stopping of the training
    print("\nTraining...")
    store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training')
    store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation')
    if plot_flag:
        # Initialize pre-trainer plot
        plt.ion()
        fig, ax_arr = plt.subplots(3)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    # Instantiate data_set
    training_dataset, validation_dataset = split_data_train_and_validation(net.params.number_of_animals, images, labels)
    # Standarize images
    training_dataset.standarize_images()
    validation_dataset.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    training_dataset.crop_images(image_size = 32)
    validation_dataset.crop_images(image_size = 32)
    # Convert labels to one hot vectors
    training_dataset.convert_labels_to_one_hot()
    validation_dataset.convert_labels_to_one_hot()
    # Reinitialize softmax and fully connected
    if first_accumulation_flag == True:
        net.reinitialize_softmax_and_fully_connected()
    # Train network
    #compute weights to be fed to the loss function (weighted cross entropy)
    net.compute_loss_weights(training_dataset.labels)
    trainer = EpochRunner(training_dataset,
                        starting_epoch = global_step,
                        print_flag = print_flag)
    validator = EpochRunner(validation_dataset,
                        starting_epoch = global_step,
                        print_flag = print_flag)
    #set criteria to stop the training
    stop_training = Stop_Training(net.params.number_of_animals,
                                check_for_loss_plateau = check_for_loss_plateau)

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

    global_step += trainer.epochs_completed
    # plot if asked
    if plot_flag:
        store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r')
        store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b')
    # store training and validation losses and accuracies
    if store_accuracy_and_error:
        store_training_accuracy_and_loss_data.save()
        store_validation_accuracy_and_loss_data.save()
    # Save network model
    net.save()
    if plot_flag:
        fig.savefig(os.path.join(net.params.save_folder,'training.pdf'))
    return global_step, net
