from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../utils')
sys.path.append('./')

import numpy as np
import h5py
import time

from blob import Blob, compute_fragment_identifier_and_blob_index
from globalfragment import give_me_list_of_global_fragments, order_global_fragments_by_distance_travelled



class BlobsListConfig(object):
    def __init__(self,number_of_animals = None, number_of_frames_per_fragment = None, number_of_frames = None, repetition = None):
        self.number_of_animals = number_of_animals
        self.number_of_frames_per_fragment = number_of_frames_per_fragment
        self.number_of_frames = number_of_frames
        self.repetition = repetition
        self.IMDB_codes = []
        self.ids_codes = []

def subsample_dataset_by_individuals(dataset, config):
    if config.number_of_animals > dataset.number_of_animals:
        raise ValueError("The number of animals for subsampling (%i) cannot be bigger than the number of animals in the dataset (%i)" %(config.number_of_animals, dataset.number_of_animals))

    if config.number_of_frames > dataset.minimum_number_of_images_per_animal:
        raise ValueError("The number of frames for subsampling (%i) cannot be bigger than the minimum number of images per animal in the dataset (%i)" %(config.number_of_frames, dataset.minimum_number_of_images_per_animal))

    # copy dataset specifics to config. This allows to restore the dataset if needed
    config.IMDB_codes = dataset.IMDB_codes
    config.ids_codes = dataset.ids_codes
    # set permutation of individuals
    np.random.seed(config.repetition)
    config.identities = dataset.identities[np.random.permutation(config.number_of_animals)]
    print("identities, ", config.identities)
    # set stating frame
    config.starting_frame = np.random.randint(dataset.minimum_number_of_images_per_animal-config.number_of_frames)
    print("starting frame, ", config.starting_frame)

    subsampled_images = []
    subsampled_centroids = []
    for identity in config.identities:
        indices_identity = np.where(dataset.labels == identity)[0]
        subsampled_images.append(np.expand_dims(dataset.images[indices_identity][config.starting_frame:config.starting_frame + config.number_of_frames], axis = 1))
        subsampled_centroids.append(np.expand_dims(dataset.centroids[indices_identity][config.starting_frame:config.starting_frame + config.number_of_frames], axis = 1))

    return np.concatenate(subsampled_images, axis = 1), np.concatenate(subsampled_centroids, axis = 1)

def generate_list_of_blobs(portraits, centroids, config):
    blobs_in_video = []
    frames_in_fragment = 0
    number_of_fragments = 0

    for frame_number, (centroids_in_frame, images_in_frame) in enumerate(zip(centroids,portraits)):
        blobs_in_frame = []
        for identity, (centroid, image) in enumerate(zip(centroids_in_frame, images_in_frame)):
            blob = Blob(centroid, None, None, None,
                        number_of_animals = config.number_of_animals)
            blob.frame_number = frame_number
            blob.portrait = image
            blob._user_generated_identity = identity

            if frame_number > 0 and frames_in_fragment <= config.number_of_frames_per_fragment + 2 and frames_in_fragment != 0:
                blob.previous = [blobs_in_video[frame_number-1][identity]]
                blobs_in_video[frame_number-1][identity].next = [blob]

            blobs_in_frame.append(blob)

        if frames_in_fragment <= config.number_of_frames_per_fragment:
            frames_in_fragment += 1
        else:
            frames_in_fragment = 0
            number_of_fragments += 1
        blobs_in_video.append(blobs_in_frame)

    return blobs_in_video

class Dataset(object):
    def __init__(self, IMDB_codes = 'A', ids_codes = 'a', cluster = 0):
        self.IMDB_codes = IMDB_codes
        self.ids_codes = ids_codes
        self.cluster = cluster

        # Get list of IMDBPaths form IMDB_codes
        print('\nReading IMDB_codes and ids_codes...')
        if not int(self.cluster):
            self.datafolder = ''
        elif int(self.cluster):
            self.datafolder = '/admin/'
        self.IMDBsDict = {
                    'A': self.datafolder + 'library/IMDBs/TU20160413_36dpf_60indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'B': self.datafolder + 'library/IMDBs/TU20160428_36dpf_60indiv_28010ImPerInd_curvaturePortrait_0.hdf5',
                    'C': self.datafolder + 'library/IMDBs/TU20160920_36dpf_64indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'D': self.datafolder + 'library/IMDBs/tu20170131_31dpf_40indiv_34770imperind_curvatureportrait_centroids_0.hdf5',
                    'E': self.datafolder + 'library/IMDBs/TU20170201_31pdf_72indiv_38739ImPerInd_curvaturePortrait_0.hdf5',
                    'F': self.datafolder + 'library/IMDBs/TU20170202_31pdf_72indiv_38913ImPerInd_curvaturePortrait_0.hdf5',
                    'a': self.datafolder + 'library/IMDBs/TU20160413_36dpf_16indiv_29938ImPerInd_curvaturePortrait_0.hdf5',
                    'b': self.datafolder + 'library/IMDBs/TU20160428_36dpf_16indiv_28818ImPerInd_curvaturePortrait_0.hdf5',
                    'd': self.datafolder + 'library/IMDBs/TU20170131_31dpf_16indiv_38989ImPerInd_curvaturePortrait_0.hdf5',
                    'c': self.datafolder + 'library/IMDBs/TU20160920_36dpf_16indiv_7731ImPerInd_curvaturePortrait_0.hdf5',
                    'e': self.datafolder + 'library/IMDBs/TU20170201_31pdf_16indiv_38997ImPerInd_curvaturePortrait_0.hdf5',
                    'f': self.datafolder + 'library/IMDBs/TU20170202_31pdf_16indiv_38998ImPerInd_curvaturePortrait_0.hdf5'
                    }
        self.IMDBPaths = []
        self.idsInIMDBs = []
        for (letter1,letter2) in zip(self.IMDB_codes,self.ids_codes):
            print('\nletter1, ', letter1)
            self.IMDBPaths.append(self.IMDBsDict[letter1])
            IMDBName = getIMDBNameFromPath(self.IMDBsDict[letter1])
            print('IMDBName, ', IMDBName)
            strain, age, numIndivIMDB, numImPerIndiv = getIMDBInfoFromName(IMDBName)
            print('numIndivIMDB', numIndivIMDB)
            print('letter2, ', letter2)
            if letter2 == 'a': # all ids
                ids = range(numIndivIMDB)
            elif letter2 == 'f': # first half idsInIMDBs
                ids = range(int(numIndivIMDB/2))
            elif letter2 == 's': # first half idsInIMDBs
                ids = range(int(numIndivIMDB/2),numIndivIMDB)
            print('ids selected, ', ids)
            self.idsInIMDBs.append(ids)
        print('IMDBPaths, ', self.IMDBPaths)
        print('idsInIMDBs, ', self.idsInIMDBs)

    def loadIMDBs(self):
        print('\n----------------------------------------------------------------')
        print('Loading images and labels form the IMDBs selected')
        # Initialize variables
        self.images = []
        self.labels = []
        self.centroids = []
        self.number_of_animals = 0
        self.strains = []
        self.ages = []
        for (IMDBPath,idsInIMDB) in zip(self.IMDBPaths,self.idsInIMDBs):
            IMDBName = getIMDBNameFromPath(IMDBPath)
            strain, age, numIndivIMDB, numImPerIndiv = getIMDBInfoFromName(IMDBName)
            print('\nExtracting imagaes from ', IMDBName)
            print('The individuals selected from this IMDB are ',  idsInIMDB)
            print('strain, ', strain)
            print('age, ', age)
            print('numIndivIMDB, ', numIndivIMDB)
            self.strains.append(strain)
            self.ages.append(age)
            # Check whether there are enough individuals in the IMDB
            if numIndivIMDB < len(idsInIMDB):
                raise ValueError('The number of indiv requested is bigger than the number of indiv in the IMDB')
            # Load IMDB
            _, imagesIMDB, labelsIMDB, centroidsIMDB, self.imsize, _, _ = loadIMDB(IMDBPath)
            # If the number of individuals requested is smaller I need to slice the IMDB
            if numIndivIMDB > len(idsInIMDB):
                imagesIMDB, labelsIMDB, centroidsIMDB = sliceDatabase(imagesIMDB, labelsIMDB, centroidsIMDB, idsInIMDB)
            # Update labels values according to the number of individuals already loaded
            labelsIMDB = labelsIMDB+self.number_of_animals
            # Append labels and images to the list
            print('images shape ', imagesIMDB.shape)
            print('labels shape ', labelsIMDB.shape)
            print('centroids shape ', centroidsIMDB.shape)
            self.images.append(imagesIMDB)
            self.labels.append(labelsIMDB)
            self.centroids.append(centroidsIMDB)
            print('The labels added are, ', np.unique(labelsIMDB))
            # Update number of individuals loaded
            self.number_of_animals += len(idsInIMDB)
            # To clear memory
            imagesIMDB = None
            labelsIMDB = None
            centroidsIMDB = None

        # Stack all images and labes
        self.images = np.concatenate(self.images, axis = 0)
        self.labels = np.concatenate(self.labels, axis = 0)
        self.centroids = np.concatenate(self.centroids, axis = 0)
        print('images shape ', self.images.shape)
        print('labels shape ', self.labels.shape)
        print('centroids shape ', self.centroids.shape)
        print('labels ', np.unique(self.labels))
        self.minimum_number_of_images_per_animal = np.min([np.sum(self.labels == i) for i in np.unique(self.labels)])
        self.identities = np.unique(self.labels)
        print('num images per label, ', self.minimum_number_of_images_per_animal)
        print('----------------------------------------------------------------\n')

''' loadDataBase '''
def loadIMDB(IMDBPath):
    # check if the train database exists, and load it!
    IMDBname = getIMDBNameFromPath(IMDBPath)
    IMDBdir = os.path.dirname(IMDBPath)
    print('\nloading %s...' %IMDBname)
    # checkDatabase(IMDBname)
    with h5py.File(IMDBdir + "/" + IMDBname + '_%i.hdf5', 'r', driver='family') as databaseTrain:
        [IMDB_info, images, labels, centroids] = getVarAttrFromHdf5(databaseTrain)
        [image_size, number_of_individual_in_IMDB, minimum_number_of_images_per_individual] = getAttrsFromGroup(IMDB_info,['imageSize', 'numIndiv','numImagesPerIndiv'])
        image_size = tuple(image_size)
        minimum_number_of_images_per_individual =  int(minimum_number_of_images_per_individual)
        print([item for item in IMDB_info.attrs.iteritems()])
    print('\ndatabase %s loaded' %IMDBname)
    return IMDB_info, images, labels, centroids, image_size, number_of_individual_in_IMDB, minimum_number_of_images_per_individual

def sliceDatabase(images, labels, centroids, individual_indices):
    ''' Select images and labels relative to a subset of individuals'''
    print('\nSlicing database...')
    sliced_images = [image for indiv_images in [images[np.where(labels == ind)[0]] for ind in individual_indices] for image in indiv_images]
    sliced_labels = [label for indiv_labels in [i*np.ones(len(np.where(labels == ind)[0])).astype(int) for i,ind in enumerate(individual_indices)] for label in indiv_labels]
    sliced_centroids = [centroid for indiv_centroids in [centroids[np.where(labels == ind)[0]] for ind in individual_indices] for centroid in indiv_centroids]
    return np.asarray(sliced_images), np.asarray(sliced_labels), np.asarray(sliced_centroids)

def getVarAttrFromHdf5(database):
    # collect the info
    groups = database.keys()
    grp = database['database']
    datanames = grp.keys()
    images = grp['images'][()]
    labels = grp['labels'][()]
    centroids = grp['centroids'][()]
    # info = [item for item in grp.attrs.iteritems()]
    return grp, images, labels, centroids

def getAttrsFromGroup(grp, variables):
    # retrieve an array from a h5py file
    return [grp.attrs[var] for var in variables]

def getIMDBNameFromPath(IMDBPath):
    filename, extension = os.path.splitext(IMDBPath)
    IMDBName = '_'.join(filename.split('/')[-1].split('_')[:-1])
    return IMDBName

def getIMDBInfoFromName(IMDBName):
    strain = IMDBName.split('_')[0]
    age = IMDBName.split('_')[1]
    numIndiv = int(IMDBName.split('_')[2][:-5])
    numImPerIndiv = int(IMDBName.split('_')[3][:-8])
    return strain, age, numIndiv, numImPerIndiv

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = Dataset(IMDB_codes = 'D', ids_codes = 'f')
    dataset.loadIMDBs()
    config = BlobsListConfig(number_of_animals = 4, number_of_frames_per_fragment = 10, number_of_frames = 34000, repetition = 2)
    portraits, centroids = subsample_dataset_by_individuals(dataset, config)

    # plt.ion()
    # plt.figure()
    # for i in range(config.number_of_animals):
    #     plt.plot(centroids[:,i,0], centroids[:,i,1])

    blobs_in_video = generate_list_of_blobs(portraits, centroids, config)
    compute_fragment_identifier_and_blob_index(blobs_in_video, config.number_of_animals)
    global_fragments = give_me_list_of_global_fragments(blobs_in_video, config.number_of_animals)
    global_fragments_ordered = order_global_fragments_by_distance_travelled(global_fragments)

    # for i in range(25):
    #     print('-------------------------')
    #     print("blob object id ",id(blobs_in_video[i][0]))
    #     print("previous object id ", id(blobs_in_video[i][0].previous), blobs_in_video[i][0].previous)
    #     print("next object id ", id(blobs_in_video[i][0].next), blobs_in_video[i][0].next)
    #     print("fragment identifier ", blobs_in_video[i][0]._fragment_identifier)
    #     print("blob index ", blobs_in_video[i][0].blob_index)
