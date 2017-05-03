from __future__ import absolute_import, division, print_function
import numpy as np

class DataSet(object):
    def __init__(self, number_of_animals = None, images = None, labels = None):
        """Create dataset of images and labels.
        param: images shaped as [num_of_images, height, width, channels]
        param: labels shaped as [num_of_labels, num_of_classes]
        """
        self.images = images
        self.labels = labels
        self.number_of_animals = number_of_animals
        #check the number of images and labels are the same. If it true set the num_images
        self.consistency_check()

    def consistency_check(self):
        if self.labels is not None:
            assert len(self.images) == len(self.labels)
            self._num_images = len(self.images)

    def standarize_images(self):
        self.images = self.images/255.
        meanIm = np.mean(self.images, axis=(1,2))
        meanIm = np.expand_dims(np.expand_dims(meanIm,axis=1),axis=2)
        stdIm = np.std(self.images,axis=(1,2))
        stdIm = np.expand_dims(np.expand_dims(stdIm,axis=1),axis=2)
        self.images = (self.images-meanIm)/stdIm

    def crop_images(self,image_size=32):
        """
        :param image_size (int): size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
        :param shift (tuple): (x,y) displacement when cropping, it can only go from -max_shift to +max_shift
        """
        current_size = self.images.shape[1]
        try:
            shift = np.divide(current_size - image_size,2)
            self.images = self.images[:,shift:current_size-shift,shift:current_size-shift,:]
        except:
            raise ValueError('The size of the input portrait must be bigger than image_size')

    # def augment_data(self): ### NOTE in the future write a class as ModelArea
    #     try:
    #         possible_shifts = get_possible_shifts_for_data_augmentation() #(0,0) is included
    #         augmented_images = []
    #         augmented_labels = []
    #         for shift in possible_shifts:
    #             new_images = crop_images(self.images,32,shift=shift)
    #             augmentedImages.append(new_images)
    #             augmentedLabels.append(self.labels)
    #         self.images = np.vstack(augmented_images)
    #         self.labels = np.expand_dims(np.hstack(augmented_labels),axis=1)
    #         self.images = crop_images(self.images)
    #     except:
    #         raise ValueError('Something went wrong during data augmentation')

    def convert_labels_to_one_hot(self):
        try:
            self.labels = dense_to_one_hot(self.labels, n_classes=self.number_of_animals)
        except:
            raise ValueError('Could not convert labels to one hot')

def split_data_train_and_validation(number_of_animals, images, labels, validation_proportion = .1):
    # Init variables
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []
    images = np.expand_dims(np.asarray(images), axis = 3)
    labels = np.expand_dims(np.asarray(labels), axis = 1)
    images, labels = shuffle_images_and_labels(images, labels)
    for i in np.unique(labels):
        # Get images of this individual
        this_indiv_images = images[np.where(labels == i)[0]]
        this_indiv_labels = labels[np.where(labels == i)[0]]
        # Compute number of images for training and validation
        num_images = len(this_indiv_labels)
        num_images_validation = np.ceil(validation_proportion*num_images).astype(int)
        num_images_training = num_images - num_images_validation

        # Get train, validation and test, images and labels
        train_images.append(this_indiv_images[:num_images_training])
        train_labels.append(this_indiv_labels[:num_images_training])
        validation_images.append(this_indiv_images[num_images_training:])
        validation_labels.append(this_indiv_labels[num_images_training:])

    train_images = np.vstack(train_images)
    train_labels = np.vstack(train_labels)
    train_images, train_labels = shuffle_images_and_labels(train_images, train_labels)
    validation_images = np.vstack(validation_images)
    validation_labels = np.vstack(validation_labels)

    return DataSet(number_of_animals, train_images, train_labels), DataSet(number_of_animals, validation_images, validation_labels)


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

# def crop_images(images,image_size=32,shift=(0,0)):
#     """
#     :param image_size (int): size of the new portrait, usually 32, since the network accepts images of 32x32  pixels
#     :param shift (tuple): (x,y) displacement when cropping, it can only go from -max_shift to +max_shift
#     """
#     current_size = images.shape[1]
#     if current_size < image_size:
#         raise ValueError('The size of the input portrait must be bigger than image_size')
#     elif current_size > image_size:
#         max_shift = np.divide(current_size - image_size,2)
#         if np.max(shift) > max_shift:
#             raise ValueError('The shift when cropping the portrait cannot be bigger than (current_size - image_size)/2')
#         images = images[:,max_shift+shift[1]:current_size-max_shift+shift[1],max_shift+shift[0]:current_size-max_shift+shift[0],:]
#         return images

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
