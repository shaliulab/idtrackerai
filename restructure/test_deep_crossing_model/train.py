from __future__ import absolute_import, division, print_function

import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np

from network import ConvNetwork

PORTRAITS_FILE = 'portraits.pkl'

class DataSet():
    def __init__(self,
               images, labels):
        """Construct a DataSet.
        Taken from tensorflow/contrib/learn/datasets/mnist.
        """
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
        assert self._num_examples == labels.shape[0]
        self.shuffle_data_and_labels()
        print("dataset with ", self._num_examples, " examples created")

    def shuffle_data_and_labels(self):
        '''Shuffles the data and the labels
        with the same permutation'''
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            print("Starting epoch", self.epochs_completed)
            self.shuffle_data_and_labels()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return (self._images[start:end], self._labels[start:end])

    @classmethod
    def from_pickled_file(cls, portraits_file, label_keyword='labels', begin_fraction=0, end_fraction=1.0):
        """Loads portraits and labels from pickled file.

        :param portraits_file: pickled file produced by produce_training_data.py
        :param label_keyword: 'labels' for noses and 'line_labels' for lines (default value is 'labels')
        :param begin_fraction:
        :param end_fraction:
        """

        with open( portraits_file,"rb") as f:
            a = pickle.load(f)
        portraits = a['portraits']
        labels = a[label_keyword]
        num = labels.shape[0]
        start = int(num*begin_fraction)
        end = int(num*end_fraction)
        return cls(portraits[start:end], labels[start:end])

if __name__ == "__main__":
    train = DataSet.from_pickled_file(PORTRAITS_FILE, 'line_labels', end_fraction=0.9)
    validation = DataSet.from_pickled_file(PORTRAITS_FILE, 'line_labels', begin_fraction=0.9)
    network = ConvNetwork()
    batch_counter = itertools.count()
    validation_counter = itertools.count()
    plt.ion()
    f, ((axp,axl),(axt1,axt2)) = plt.subplots(2,2)

    while True:
        if next(batch_counter)%40 == 0:
            (image,target_label) = validation.next_batch(1)
            label = network.prediction(image)
            axp.imshow(np.squeeze(image))
            axl.imshow(np.squeeze(label))
            axt1.imshow(0.75 < np.squeeze(label))
            axt2.imshow(0.9 < np.squeeze(label))
            plt.draw()
            plt.pause(0.0001)
            network.save('/tmp/checkpoint', next(validation_counter))
        network.train(train.next_batch(100))
