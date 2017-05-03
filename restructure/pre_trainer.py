from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./network')

import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from network_params import NetworkParams
from get_data import DataSet, split_data_train_and_validation
from id_CNN import ConvNetwork
from globalfragment import get_images_and_labels_from_global_fragment, give_me_pre_training_global_fragments
from epoch_runner import EpochRunner
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

def pre_train(pretraining_global_fragments, number_of_global_fragments, params, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag, plot_flag):
    global_epoch = 0
    net = ConvNetwork(params)
    # Save accuracy and error during training and validation
    # The loss and accuracy of the validation are saved to allow the automatic stopping of the training
    store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training')
    store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation')
    if plot_flag:
        # Initialize pre-trainer plot
        plt.ion()
        fig, ax_arr = plt.subplots(3)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        epoch_index_to_plot = 0
    # Start loop for pre training in the global fragments
    for i, pretraining_global_fragment in enumerate(tqdm(pretraining_global_fragments, desc = 'Pretraining network')):
        # Get images and labels from the current global fragment
        images, labels = get_images_and_labels_from_global_fragment(pretraining_global_fragment)
        # Instantiate data_set
        training_dataset, validation_dataset = split_data_train_and_validation(images,labels)
        # Standarize images
        training_dataset.standarize_images()
        validation_dataset.standarize_images()
        # Crop images from 36x36 to 32x32 without performing data augmentation
        training_dataset.crop_images(image_size = 32)
        validation_dataset.crop_images(image_size = 32)
        # Convert labels to one hot vectors
        training_dataset.convert_labels_to_one_hot()
        validation_dataset.convert_labels_to_one_hot()
        # Restore network
        net.restore()
        # Train network
        #compute weights to be fed to the loss function (weighted cross entropy)
        net.compute_loss_weights(data._train_labels)
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
            # --- Training
            feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, net.train)
            ### NOTE here we can shuffle the training data if we think it is necessary.
            # --- Validation
            feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, net.validate)
            # update global step
            net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed))
            # write summaries if asked
            if save_summaries:
                net.write_summaries(trainer.starting_epoch + trainer._epochs_completed,feed_dict_train, feed_dict_val)
            # Update counter
            trainer._epochs_completed += 1


        # plot if asked
        if plot_flag:
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
        fig.savefig(os.path.join(net.params.save_folder,'pretraining.pdf'))




if __name__ == "__main__":
    import numpy as np
    global_fragments = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/preprocessing/global_fragments.npy')
    #network parameters
    video = np.load('/home/chronos/Desktop/IdTrackerDeep/videos/8zebrafish_conflicto/preprocessing/video_object.npy').item()
    learning_rate = 0.01
    keep_prob = 1.0
    use_adam_optimiser = False
    scopes_layers_to_optimize = None
    restore_folder = None
    save_folder = './pretraining'
    knowledge_transfer_folder = './pretraining'
    params = NetworkParams(video,learning_rate, keep_prob,use_adam_optimiser, scopes_layers_to_optimize,restore_folder , save_folder , knowledge_transfer_folder)
    pre_train(global_fragments, params)
