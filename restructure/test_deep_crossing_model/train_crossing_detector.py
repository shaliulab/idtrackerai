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
    def __init__(self, video_folder, training_dataset, validation_dataset, num_epochs = 50, plot_flag = True):
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
        self.ckpt_path = os.path.join(video_folder, 'deep_crossings_ckpt')
        self.plot_flag = plot_flag
        self.net = ConvNetwork(weight_positive = training_dataset.weight_positive, architecture = cnn_model_crossing_detector, learning_rate = 0.001)
        self.train_model()

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
            if self.plot_flag:
                self.plotter(loss_train, acc_train, loss_val, acc_val)

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
    video_folder = video._video_folder
    training_file_in_video_folder = [f for f in listdir(video_folder) if isfile(join(video_folder, f)) and '.npy' in f and 'training' in f]
    if len(training_file_in_video_folder) > 0:
        training_set = np.load(training_file_in_video_folder[0]).item()
        validation_file_in_video_folder = [f for f in listdir(video_folder) if isfile(join(video_folder, f)) and '.npy' in f and 'validation' in f]
        if len(training_file_in_video_folder) > 0:
            validation_set = np.load(validation_file_in_video_folder[0]).item()
        else:
            raise ValueError("No validation found, silly moose!")
        test_file_in_video_folder = [f for f in listdir(video_folder) if isfile(join(video_folder, f)) and '.npy' in f and 'test' in f]
        if len(training_file_in_video_folder) > 0:
            test_set = np.load(test_file_in_video_folder[0]).item()
        else:
            raise ValueError("No test found, silly moose!")
    else:
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

    crossing_detector = TrainDeepCrossing(video_folder, training_set, validation_set, num_epochs = 100, plot_flag = True)
    test_images = test_set.generate_test_images()
    predictions = crossing_detector.net.prediction(test_images)
    print(predictions)
    fish_indices = np.where(predictions[:,0]>predictions[:,1])[0]
    crossing_indices = np.where(predictions[:,0]<predictions[:,1])[0]
    print("accuracy on test: ", len(crossing_indices)/(len(crossing_indices)+len(fish_indices)))
