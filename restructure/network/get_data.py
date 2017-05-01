from __future__ import absolute_import, division, print_function
import numpy as np

class Get_Data(object):
    def __init__(self, images, labels, validation_proportion = .1, augment_data = False):
        self.images = np.expand_dims(np.asarray(images),axis=3)
        self.labels = np.expand_dims(np.asarray(labels),axis=1)
        self.number_of_animals = len(np.unique(self.labels))
        self.num_images = len(self.images)
        self.validation_proportion = validation_proportion
        self.augment_data_flag = augment_data
        self._train_images = []
        self._train_labels = []
        self._validation_images = []
        self._validation_labels = []

    def standarize_images(self):
        self.images = self.images/255.
        meanIm = np.mean(self.images, axis=(1,2))
        meanIm = np.expand_dims(np.expand_dims(meanIm,axis=1),axis=2)
        stdIm = np.std(self.images,axis=(1,2))
        stdIm = np.expand_dims(np.expand_dims(stdIm,axis=1),axis=2)
        self.images = (self.images-meanIm)/stdIm

    def split_train_and_validation(self):
        self.images, self.labels = shuffle_images_and_labels(self.images, self.labels)
        for i in np.unique(self.labels):
            # Get images of this individual
            this_indiv_images = self.images[np.where(self.labels == i)[0]]
            this_indiv_labels = self.labels[np.where(self.labels == i)[0]]
            # Compute number of images for training and validation
            num_images = len(this_indiv_labels)
            num_images_validation = np.ceil(self.validation_proportion*num_images).astype(int)
            num_images_training = num_images - num_images_validation

            # Get train, validation and test, images and labels
            self._train_images.append(this_indiv_images[:num_images_training])
            self._train_labels.append(this_indiv_labels[:num_images_training])
            self._validation_images.append(this_indiv_images[num_images_training:])
            self._validation_labels.append(this_indiv_labels[num_images_training:])

        self._train_images = np.vstack(self._train_images)
        self._train_labels = np.vstack(self._train_labels)
        self._train_images, self._train_labels = shuffle_images_and_labels(self._train_images, self._train_labels)
        self._validation_images = np.vstack(self._validation_images)
        self._validation_labels = np.vstack(self._validation_labels)
        self._num_train_images = len(self._train_images)
        self._num_validation_images = len(self._validation_labels)

    def crop_images_and_augment_data(self):
        if not self.augment_data_flag:
            self._train_images = crop_images(self._train_images)
            self._validation_images = crop_images(self._validation_images)
        else:
            self.augment_data()

    def augment_data(self): ### NOTE in the future write a class as ModelArea
        possible_shifts = get_possible_shifts_for_data_augmentation() #(0,0) is included
        augmented_images = []
        augmented_labels = []
        for shift in possible_shifts:
            new_images = crop_images(self._train_images,32,shift=shift)
            augmentedImages.append(new_images)
            augmentedLabels.append(self._train_labels)
        self._train_images = np.vstack(augmented_images)
        self._train_labels = np.expand_dims(np.hstack(augmented_labels),axis=1)
        self._train_images = crop_images(self._train_images)

    def convert_labels_to_one_hot(self):
        self._train_labels = dense_to_one_hot(self._train_labels, n_classes=self.number_of_animals)
        self._validation_labels = dense_to_one_hot(self._validation_labels, n_classes=self.number_of_animals)


def shuffle_images_and_labels(images, labels):
    """Shuffles images and labels with a random
    permutation, according to the number of examples"""
    perm = np.random.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]
    return images, labels

def get_possible_shifts_for_data_augmentation():
    possibleShifts = []
    possibleShifts.append(list(itertools.combinations_with_replacement(range(-2,3),2)))
    possibleShifts.append(list(itertools.permutations(range(-2,3),2)))
    possibleShifts.append(list(itertools.combinations(range(-2,3),2)))
    possibleShifts = [shift for l in possibleShifts for shift in l]
    possibleShifts = set(possibleShifts)
    return possibleShifts

def crop_images(images,image_size=32,shift=(0,0)):
    """
    :param image_size (int): size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
    :param shift (tuple): (x,y) displacement when cropping, it can only go from -max_shift to +max_shift
    """
    current_size = images.shape[1]
    if current_size < image_size:
        raise ValueError('The size of the input portrait must be bigger than image_size')
    elif current_size > image_size:
        max_shift = np.divide(current_size - image_size,2)
        if np.max(shift) > max_shift:
            raise ValueError('The shift when cropping the portrait cannot be bigger than (current_size - image_size)/2')
        images = images[:,max_shift+shift[1]:current_size-max_shift+shift[1],max_shift+shift[0]:current_size-max_shift+shift[0],:]
        return images

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
