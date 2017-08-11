from __future__ import absolute_import, division, print_function

import itertools
import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../network')
from deep_crossing_model import ConvNetwork
from cnn_architectures import cnn_model_crossing_detector
import seaborn as sns

class TrainDeepCrossing(object):
    def __init__(self, session_folder, training_dataset, validation_dataset, num_epochs = 50, plot_flag = True):
        """Build the dataset and trains the model
        The dataset is built according to
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self._train_images = training_dataset.images
        self._train_labels = training_dataset.labels
        self._validation_images = validation_dataset.images
        self._validation_labels = validation_dataset.labels
        self._epoches_completed = 0
        self._index_in_epoch = 0
        self._index_in_epoch_val = 0
        self._num_examples_train = len(training_dataset.images)
        self._num_examples_val = len(validation_dataset.images)
        self.num_epochs = num_epochs
        self.plot_flag = plot_flag
        self.save_path(session_folder)
        self.net = ConvNetwork(weight_positive = training_dataset.weight_positive,
                                architecture = cnn_model_crossing_detector,
                                learning_rate = 0.001,
                                image_size = training_dataset.images.shape[1:])
        self.train_model()

    def save_path(self, session_folder):
        print('setting path to save crossing detector model')
        self.crossing_detector_path = os.path.join(session_folder, 'crossing_detector')
        if not os.path.isdir(self.crossing_detector_path):
            print('creating folder to store checkpoints for the crossing_detector')
            os.makedirs(self.crossing_detector_path)

    def next_batch_train(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples_train:
            # Finished epoch
            self._epoches_completed += 1
            print("***Starting epoch", self._epoches_completed)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples_train
        end = self._index_in_epoch
        return (self._train_images[start:end], self._train_labels[start:end])

    # def next_batch_validation(self, batch_size):
    #     """Return the next `batch_size` examples from this data set."""
    #     start = self._index_in_epoch_val
    #     self._index_in_epoch_val += batch_size
    #     end = self._index_in_epoch_val
    #     return (self._validation_images[start:end], self._validation_labels[start:end])

    def next_batch_validation(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch_val
        self._index_in_epoch_val += batch_size
        if self._index_in_epoch_val > self._num_examples_val:
            # Finished epoch
            self._epoches_completed += 1
            print("***Starting epoch", self._epoches_completed)
            # Start next epoch
            start = 0
            self._index_in_epoch_val = batch_size
            assert batch_size <= self._num_examples_val
        end = self._index_in_epoch_val
        return (self._validation_images[start:end], self._validation_labels[start:end])


    def train_model(self):
        batch_counter = itertools.count()
        # validation_counter = itertools.count()
        if self.plot_flag:
            plt.ion()
            sns.set_style("ticks")
            self.fig, self.ax_arr = plt.subplots(2,1)

        loss_train = []
        acc_train = []
        loss_val = []
        acc_val = []
        while self._epoches_completed < self.num_epochs:
            #train network
            print("epoch: ", self._epoches_completed)
            loss, acc = self.net.train(self.next_batch_train(batch_size = 100))
            loss_train.append(loss)
            acc_train.append(acc)
            loss, acc = self.net.validate(self.next_batch_validation(batch_size = 100))
            loss_val.append(loss)
            acc_val.append(acc)
            if next(batch_counter)%100 == 0:
                #save checkpoint
                self.net.save(os.path.join(self.crossing_detector_path, 'checkpoint'), next(batch_counter))
            if self.plot_flag:
                self.plotter(loss_train, acc_train, loss_val, acc_val)

    def next_batch_test(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.number_of_image_predicted
        self.number_of_image_predicted += batch_size
        end = self.number_of_image_predicted
        return self.test_images[start:end]

    def predict(self, test_images):
        self.test_images = test_images
        self.number_of_image_predicted = 0
        predictions = []
        while self.number_of_image_predicted < len(test_images):
            predictions.extend(self.net.prediction(self.next_batch_test(batch_size = 100)))

        return predictions


    def plotter(self, loss_train, acc_train, loss_val, acc_val):
        ax = self.ax_arr[0]
        ax.clear()
        ax.plot(loss_train, 'r')
        ax.plot(loss_val, 'b')
        ax.set_xlim([0,self.num_epochs])
        ax.set_ylabel('loss')

        ax = self.ax_arr[1]
        ax.clear()
        ax.plot(acc_train, 'r')
        ax.plot(acc_val, 'b')
        ax.set_xlim([0,self.num_epochs])
        ax.set_ylabel('accuracy')
        ax.set_xlabel('epochs')

        plt.show()

if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join
    import sys
    sys.setrecursionlimit(1000000)
    sys.path.append('../')
    sys.path.append('../utils')

    from GUI_utils import selectDir
    from blob import ListOfBlobs, Blob
    from deep_crossings import CrossingDataset

    ''' select blobs list tracked to compare against ground truth '''
    session_path = selectDir('./') #select path to video
    video_path = os.path.join(session_path,'video_object.npy')
    print("loading video object...")
    video = np.load(video_path).item(0)
    session_folder = video._session_folder

    blobs_path = video.blobs_path
    global_fragments_path = video.global_fragments_path
    list_of_blobs = ListOfBlobs.load(blobs_path)
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
