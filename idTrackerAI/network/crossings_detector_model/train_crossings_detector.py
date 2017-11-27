from __future__ import absolute_import, division, print_function

import itertools
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import psutil

sys.path.append('./network/crossings_detector_model')
from cnn_architectures import cnn_model_crossing_detector
from network_params_crossings import NetworkParams_crossings
from crossings_detector_model import ConvNetwork_crossings
from stop_training_criteria_crossings import Stop_Training
from store_accuracy_and_loss_crossings import Store_Accuracy_and_Loss
from epoch_runner_crossings import EpochRunner

class TrainDeepCrossing(object):
    def __init__(self, net, training_dataset, validation_dataset, num_epochs = 50, plot_flag = True):
        """Build the dataset and trains the model
        The dataset is built according to
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.num_epochs = num_epochs
        self.plot_flag = plot_flag
        self.net = net
        self.train_model()

    def train_model(self, global_step = 0,
                            check_for_loss_plateau = None,
                            print_flag = True,
                            store_accuracy_and_error = False):
        print("\nTraining...")
        store_training_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'training')
        store_validation_accuracy_and_loss_data = Store_Accuracy_and_Loss(self.net, name = 'validation')

        if self.plot_flag:
            # Initialize pre-trainer plot
            plt.ion()
            fig, ax_arr = plt.subplots(3)
            fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

        self.net.compute_loss_weights(self.training_dataset.labels)
        trainer = EpochRunner(self.training_dataset,
                            starting_epoch = global_step,
                            print_flag = print_flag)
        validator = EpochRunner(self.validation_dataset,
                            starting_epoch = global_step,
                            print_flag = print_flag)
        #set criteria to stop the training
        stop_training = Stop_Training(check_for_loss_plateau = check_for_loss_plateau)
        print("entering the epochs loop...")
        while not stop_training(store_training_accuracy_and_loss_data,
                                store_validation_accuracy_and_loss_data,
                                trainer._epochs_completed):
            # --- Training
            feed_dict_train = trainer.run_epoch('Training', store_training_accuracy_and_loss_data, self.net.train)
            # --- Validation
            feed_dict_val = validator.run_epoch('Validation', store_validation_accuracy_and_loss_data, self.net.validate)
            # update global step
            # net.session.run(net.global_step.assign(trainer.starting_epoch + trainer._epochs_completed))
            # write summaries if asked
            # if save_summaries:
            #     net.write_summaries(trainer.starting_epoch + trainer._epochs_completed,feed_dict_train, feed_dict_val)
            # Update counter
            trainer._epochs_completed += 1
            validator._epochs_completed += 1

        global_step += trainer.epochs_completed
        print('\nvalidation losses: ', store_validation_accuracy_and_loss_data.loss)
        # plot if asked
        if self.plot_flag:
            store_training_accuracy_and_loss_data.plot(ax_arr, color = 'r')
            store_validation_accuracy_and_loss_data.plot(ax_arr, color ='b')
        # store training and validation losses and accuracies
        if store_accuracy_and_error:
            store_training_accuracy_and_loss_data.save()
            store_validation_accuracy_and_loss_data.save()
        # Save network model
        self.net.save(global_step = global_step)
        if self.plot_flag:
            fig.savefig(os.path.join(self.net.params.save_folder,'crossing_detector.pdf'))

    # def next_batch_test(self, batch_size):
    #     """Return the next `batch_size` examples from this data set."""
    #     start = self.number_of_image_predicted
    #     self.number_of_image_predicted += batch_size
    #     end = self.number_of_image_predicted
    #     return self.test_images[start:end]
    #
    # def predict(self, test_images):
    #     self.test_images = test_images
    #     self.number_of_image_predicted = 0
    #     predictions = []
    #     while self.number_of_image_predicted < len(test_images):
    #         predictions.extend(self.net.prediction(self.next_batch_test(batch_size = 100)))
    #
    #     return predictions
    #
    # def get_all_predictions(self, test_set):
    #     # compute maximum number of images given the available RAM
    #     image_size_bytes = np.prod(test_set.image_size**2)*4
    #     number_of_images_to_be_fitted_in_RAM = len(test_set.test)
    #     num_images_that_can_fit_in_RAM = int(psutil.virtual_memory().available*.9/image_size_bytes)
    #     if number_of_images_to_be_fitted_in_RAM > num_images_that_can_fit_in_RAM:
    #         print("There is NOT enough RAM to host %i images" %number_of_images_to_be_fitted_in_RAM)
    #         number_of_predictions_retrieved = 0
    #         predictions = []
    #         i = 0
    #         while number_of_predictions_retrieved < number_of_images_to_be_fitted_in_RAM:
    #             images = test_set.generate_test_images(interval = (i*num_images_that_can_fit_in_RAM, (i+1)*num_images_that_can_fit_in_RAM))
    #             predictions.extend(self.predict(images))
    #             number_of_predictions_retrieved = len(predictions)
    #             i += 1
    #     else:
    #         test_images = test_set.generate_test_images()
    #         predictions = self.predict(test_images)
    #     return predictions

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import sys

    sys.path.append('../')
    sys.path.append('../utils')

    from GUI_utils import selectDir
    from list_of_blobs import ListOfBlobs
    from blob import Blob
    from deep_crossings import CrossingDataset

    ''' select blobs list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    session_folder = video.session_folder

    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(video, blobs_path)
    blobs = list_of_blobs.blobs_in_video

    training_set = CrossingDataset(blobs, video)
    training_set.get_data(sampling_ratio_start = 0, sampling_ratio_end = .9, scope = 'training')
    # np.save(os.path.join(video_folder, '_training_set.npy'), training_set)
    validation_set = CrossingDataset(blobs, video)
    validation_set.get_data(sampling_ratio_start = .9, sampling_ratio_end = 1., scope = 'validation')
    # np.save(os.path.join(video_folder, '_validation_set.npy'), validation_set)
    test_set = CrossingDataset(blobs, video)
    # np.save(os.path.join(video_folder,'_test_set.npy'), validation_set)

    crossing_detector = TrainDeepCrossing(session_folder, training_set, validation_set, num_epochs = 95, plot_flag = True)
    test_images = test_set.generate_test_images()
    softmax_probs, predictions = crossing_detector.net.prediction(test_images)
    print(predictions)
    fish_indices = np.where(predictions == 0)[0]
    crossing_indices = np.where(predictions == 1)[0]
    print("accuracy on test: ", len(crossing_indices)/(len(crossing_indices)+len(fish_indices)))
