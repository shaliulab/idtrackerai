from __future__ import absolute_import, division, print_function
import sys
sys.path.append('./network')

import itertools
from tqdm import tqdm

from cnn_config import Network_Params
from get_data import Get_Data
from id_CNN import ConvNetwork
from globalfragment import get_images_and_labels_from_global_fragment, give_me_pre_training_global_fragments
from train_id_CNN import TrainIdCNN

def pre_train(global_fragments, params, store_accuracy_and_error, check_for_loss_plateau, save_summaries, print_flag):
    # get global equispaced global fragments along the video to pretrain the network
    pretraining_global_fragments = give_me_pre_training_global_fragments(global_fragments)

    global_epoch = 0
    net = ConvNetwork(params)
    for i, pretraining_global_fragment in enumerate(tqdm(pretraining_global_fragments, desc = 'Pretraining network')):
        # Get images and labels from the current global fragment
        images, labels = get_images_and_labels_from_global_fragment(pretraining_global_fragment)
        # Instantiate data_set
        data_set = Get_Data(images,labels, augment_data = False)
        # Standarize images
        data_set.standarize_images()
        # Split data_set in _train_images, _train_labels, _train_labels, _validation_labels
        data_set.split_train_and_validation()
        # Crop images from 36x36 to 32x32 without performing data augmentation
        data_set.crop_images_and_augment_data()
        # Convert labels to one hot vectors
        data_set.convert_labels_to_one_hot()
        # Restore network
        net.restore()
        # Train network
        trainer = TrainIdCNN(net,
                            data_set,
                            starting_epoch = global_epoch,
                            save_summaries = save_summaries,
                            store_accuracy_and_error = store_accuracy_and_error,
                            check_for_loss_plateau = check_for_loss_plateau,
                            print_flag = print_flag)
        # We start the training
        trainer.train_model()
        # Update global_epoch counter
        global_epoch += trainer._epoches_completed
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
