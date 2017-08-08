from __future__ import absolute_import, division, print_function

import itertools
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from network_nose_detector import ConvNetwork

class TrainNoseDetector(object):
    def __init__(self, video_path, num_epochs = 50, data_dict = {}, label_keyword = 'labels', plot_flag = True):
        """Build the dataset and trains the model
        The dataset is built according to
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self._epoches_completed = 0
        self._index_in_epoch = 0
        self._index_in_epoch_val = 0
        self._num_examples = 0
        self.num_epochs = num_epochs
        self.data_dict = data_dict
        self.label_keyword = label_keyword
        self.plot_flag = plot_flag
        self.get_data_path(video_path)
        self.net = ConvNetwork()
        self.train_model()

    def get_data_path(self, video_path):
        print('setting path to dataset')
        video_folder = os.path.dirname(video_path)
        self.nose_detector_ckp_path = os.path.join(video_folder, 'nose_detector_checkpoints')
        if not os.path.isdir(self.nose_detector_ckp_path):
            print('creating folder to store checkpoints')
            os.makedirs(self.nose_detector_ckp_path)

        self._data_path = os.path.join(video_folder, 'preprocessing/nose_detector_dataset.pkl')

    def shuffle_data_and_labels(self, shuffle_only_train = False):
        '''Shuffles the data and the labels with the same permutation
        if shuffle_only_train = True it permutes only the training set of images,
        else it permutes all the images and spits them into training and validation'''
        if not shuffle_only_train:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self.get_train_and_validation_data()
        else:
            perm = np.arange(self._num_train_examples)
            np.random.shuffle(perm)
            self._train_images = self._train_images[perm]
            self._train_labels = self._train_labels[perm]

    def next_batch_train(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_train_examples:
            # Finished epoch
            self._epoches_completed += 1
            print("***Starting epoch", self._epoches_completed)
            self.shuffle_data_and_labels(shuffle_only_train = True)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return (self._train_images[start:end], self._train_labels[start:end])

    def next_batch_validation(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch_val
        self._index_in_epoch_val += batch_size
        end = self._index_in_epoch_val
        return (self._validation_images[start:end], self._validation_labels[start:end])

    def load_data_pickle(self):
        if not self.data_dict:
            self.data_dict = pickle.load(open(self._data_path,'rb'))
        print("Training on ", self.label_keyword)
        self._images = self.data_dict['images']
        self._labels = self.data_dict[self.label_keyword]
        self._num_examples = len(self._labels)
        print("Dataset with ", self._num_examples, " samples has been loaded")
        self.shuffle_data_and_labels()

    def slice_dataset(self, begin_fraction = 0, end_fraction = 1.0):
        start = int(self._num_examples * begin_fraction)
        end = int(self._num_examples * end_fraction)
        sliced_images = self._images[start : end]
        sliced_labels = self._labels[start : end]
        return sliced_images, sliced_labels

    def get_train_and_validation_data(self):
        self._train_images, self._train_labels = self.slice_dataset(end_fraction = 0.9)
        self._num_train_examples = len(self._train_labels)
        self._validation_images, self._validation_labels = self.slice_dataset(begin_fraction = 0.9)

    def train_model(self):
        self.load_data_pickle()
        batch_counter = itertools.count()
        validation_counter = itertools.count()
        plt.ion()
        if self.plot_flag:
            f, ((axp,axl),(axt1,axt2)) = plt.subplots(2,2)

        while self._epoches_completed < self.num_epochs:
            if next(batch_counter)%40 == 0:
                if self.plot_flag:
                    (image,target_label) = self.next_batch_validation(batch_size = 1)
                    label = self.net.prediction(image)
                    axp.imshow(np.squeeze(image))
                    axl.imshow(np.squeeze(label))
                    axt1.imshow(0.75 < np.squeeze(label))
                    axt2.imshow(0.9 < np.squeeze(label))
                    plt.draw()
                    plt.pause(0.0001)
                #save checkpoint
                self.net.save(os.path.join(self.nose_detector_ckp_path, 'checkpoint'), next(validation_counter))
            #train network
            self.net.train(self.next_batch_train(batch_size = 100))

if __name__ == "__main__":
    video_path = '/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Conflicto8/conflict3and4_20120316T155032_1.avi'
    label_keyword = 'line_labels'
    data = TrainNoseDetector(video_path, num_epochs = 2, label_keyword = label_keyword, plot_flag = True)
