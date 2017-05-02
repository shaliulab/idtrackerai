from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./network')

import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn_config import Network_Params
from get_data import GetData, DataSet
from id_CNN import ConvNetwork
from globalfragment import get_images_and_labels_from_global_fragment, give_me_pre_training_global_fragments
from train_id_CNN import TrainIdCNN
from stop_training_criteria import Stop_Training
from store_accuracy_and_loss import Store_Accuracy_and_Loss

def pre_train(global_fragments, number_of_global_fragments, params, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag):
    # get global equispaced global fragments along the video to pretrain the network
    pretraining_global_fragments = give_me_pre_training_global_fragments(global_fragments, number_of_global_fragments = number_of_global_fragments)

    global_epoch = 0
    net = ConvNetwork(params)
    for i, pretraining_global_fragment in enumerate(tqdm(pretraining_global_fragments, desc = 'Pretraining network')):
        # Get images and labels from the current global fragment
        images, labels = get_images_and_labels_from_global_fragment(pretraining_global_fragment)
        # Instantiate data_set
        data = GetData(images,labels, augment_data = False)
        # Standarize images
        data.standarize_images()
        # Split data_set in _train_images, _train_labels, _train_labels, _validation_labels
        data.split_train_and_validation()
        # Crop images from 36x36 to 32x32 without performing data augmentation
        data.crop_images_and_augment_data()
        # Convert labels to one hot vectors
        data.convert_labels_to_one_hot()
        #create the training and validation dataset
        training_dataset = DataSet(data._train_images, data._train_labels)
        validation_dataset = DataSet(data._validation_images, data._validation_labels)
        # Restore network
        net.restore()
        # Train network
        #compute weights to be fed to the loss function (weighted cross entropy)
        net.compute_loss_weights(data._train_labels)

        trainer = TrainIdCNN(net,
                            training_dataset,
                            starting_epoch = global_epoch,
                            check_for_loss_plateau = check_for_loss_plateau,
                            print_flag = print_flag)

        validator = TrainIdCNN(net,
                            validation_dataset,
                            starting_epoch = global_epoch,
                            check_for_loss_plateau = check_for_loss_plateau,
                            print_flag = print_flag)
        #set criteria to stop the training
        stop_training = Stop_Training(trainer.num_epochs,
                                    net.params.number_of_animals,
                                    check_for_loss_plateau = trainer.check_for_loss_plateau)
        # Save accuracy and error during training and validation
        # The loss and accuracy of the validation are saved to allow the automatic stopping of the training
        training_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'training')
        validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(net, name = 'validation')
        plt.ion()
        fig, ax_arr = plt.subplots(2, sharex=True)
        while not stop_training(training_accuracy_and_loss_data,
                                validation_accuracy_and_loss_data,
                                trainer._epochs_completed):
            # --- Training
            feed_dict_train = trainer.run_epoch('Training', training_accuracy_and_loss_data, net.train)
            ### NOTE here we can shuffle the training data if we think it is necessary.
            # --- Validation
            feed_dict_val = validator.run_epoch('Validation', validation_accuracy_and_loss_data, net.validate)
            # update global step
            net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed)) # set and update(eval) global_step with index, i
            # take times (for library)


            # write summaries if asked
            if save_summaries:
                net.write_summaries(trainer.starting_epoch + trainer._epochs_completed,feed_dict_train, feed_dict_val)
            # Update counter
            trainer._epochs_completed += 1

        # plot if asked
        training_accuracy_and_loss_data.plot(ax_arr)
        validation_accuracy_and_loss_data.plot(ax_arr)


        if store_accuracy_and_error:
            training_accuracy_and_loss_data.save()
            validation_accuracy_and_loss_data.save()
        # Update global_epoch counter
        global_epoch += trainer._epochs_completed
        # Save network model
        net.save()
        # Plot training
        # trainer.plot()



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
    params = Network_Params(video,learning_rate, keep_prob,use_adam_optimiser, scopes_layers_to_optimize,restore_folder , save_folder , knowledge_transfer_folder)
    pre_train(global_fragments, params)
