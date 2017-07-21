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
from scipy.stats import truncnorm

""" generage jobConfig """
class LibraryJobConfig(object):
    def __init__(self,cluster = None, test_dictionary = None):
        self.cluster = int(cluster)
        for key in test_dictionary:
            setattr(self, key, test_dictionary[key])

    def create_folders_structure(self):
        #create main condition folder
        self.condition_path = os.path.join('./library','library_test_' + self.test_name)
        if not os.path.exists(self.condition_path):
            os.makedirs(self.condition_path)
        #create subfolders for group sizes
        for group_size in self.group_sizes:
            group_size_path = os.path.join(self.condition_path,'group_size_' + str(group_size))
            if not os.path.exists(group_size_path):
                os.makedirs(group_size_path)
            #create subfolders for frames_in_video
            for frames_in_video in self.frames_in_video:
                num_frames_path = os.path.join(group_size_path,'num_frames_' + str(frames_in_video))
                if not os.path.exists(num_frames_path):
                    os.makedirs(num_frames_path)
                #create subfolders for frames_in_fragment
                for frames_in_fragment in self.frames_per_individual_fragment:
                    frames_in_fragment_path = os.path.join(num_frames_path, 'frames_in_fragment_' + str(frames_in_fragment))
                    if not os.path.exists(frames_in_fragment_path):
                        os.makedirs(frames_in_fragment_path)
                    for repetition in self.repetitions:
                        repetition_path = os.path.join(frames_in_fragment_path, 'repetition_' + str(repetition))
                        if not os.path.exists(repetition_path):
                            os.makedirs(repetition_path)

def check_if_repetition_has_been_computed(results_data_frame, job_config, group_size, frames_in_video, frames_in_fragment, repetition):

    return len(results_data_frame.query('test_name == @job_config.test_name' +
                                            ' & CNN_model == @job_config.CNN_model' +
                                            ' & knowledge_transfer_flag == @job_config.knowledge_transfer_flag' +
                                            ' & knowledge_transfer_folder == @job_config.knowledge_transfer_folder' +
                                            ' & pretraining_flag == @job_config.pretraining_flag' +
                                            ' & percentage_of_frames_in_pretaining == @job_config.percentage_of_frames_in_pretaining' +
                                            ' & only_accumulate_one_fragment == @job_config.only_accumulate_one_fragment' +
                                            ' & train_filters_in_accumulation == @job_config.train_filters_in_accumulation' +
                                            ' & accumulation_certainty == @job_config.accumulation_certainty' +
                                            ' & IMDB_codes == @job_config.IMDB_codes' +
                                            ' & ids_codes == @job_config.ids_codes' +
                                            ' & group_size == @group_size' +
                                            ' & frames_in_video == @frames_in_video' +
                                            ' & frames_per_fragment == @frames_in_fragment' +
                                            ' & repetition == @repetition')) != 0

""" generate blob lists """
class BlobsListConfig(object):
    def __init__(self,
                    number_of_animals = None,
                    number_of_frames_per_fragment = None,
                    std_number_of_frames_per_fragment = None,
                    number_of_frames = None,
                    repetition = None):
        self.number_of_animals = number_of_animals
        self.number_of_frames_per_fragment = number_of_frames_per_fragment
        self.std_number_of_frames_per_fragment = std_number_of_frames_per_fragment
        self.max_number_of_frames_per_fragment = number_of_frames
        self.min_number_of_frames_per_fragment = 1
        self.number_of_frames = number_of_frames
        self.repetition = repetition
        self.IMDB_codes = []
        self.ids_codes = []

def subsample_dataset_by_individuals(dataset, config):
    # We need to consider that for every individual fragment there are two frames that are not considered for training
    # In order to have the correct number of trainable images per individual fragment we need to increase the number of
    # frames in the video to account for those two frames. The total number of frames needes is:
    # number_of_frames = int(config.number_of_frames + 2 * config.number_of_frames / config.number_of_frames_per_fragment)
    number_of_frames = config.number_of_frames
    if config.number_of_animals > dataset.number_of_animals:
        raise ValueError("The number of animals for subsampling (%i) cannot be bigger than the number of animals in the dataset (%i)" %(config.number_of_animals, dataset.number_of_animals))

    if number_of_frames > dataset.minimum_number_of_images_per_animal:
        raise ValueError("The number of frames for subsampling (%i) cannot be bigger than the minimum number of images per animal in the dataset (%i)" %(number_of_frames, dataset.minimum_number_of_images_per_animal))

    # copy dataset specifics to config. This allows to restore the dataset if needed
    config.IMDB_codes = dataset.IMDB_codes
    config.ids_codes = dataset.ids_codes
    # set permutation of individuals
    np.random.seed(config.repetition)
    config.identities = dataset.identities[np.random.permutation(config.number_of_animals)]
    print("identities, ", config.identities)
    # set stating frame
    # we set the starting frame so that we take images from both videos of the library.
    # the greates unbalance can be 1/3 from video_1 and 2/3 from video_2
    config.starting_frame = np.random.randint(dataset.minimum_number_of_images_per_animal-number_of_frames)
    print("starting frame, ", config.starting_frame)

    subsampled_images = []
    subsampled_centroids = []
    for identity in config.identities:
        indices_identity = np.where(dataset.labels == identity)[0]
        subsampled_images.append(np.expand_dims(dataset.images[indices_identity][config.starting_frame:config.starting_frame + number_of_frames], axis = 1))
        subsampled_centroids.append(np.expand_dims(dataset.centroids[indices_identity][config.starting_frame:config.starting_frame + number_of_frames], axis = 1))

    return np.concatenate(subsampled_images, axis = 1), np.concatenate(subsampled_centroids, axis = 1)

def get_next_number_of_frames_in_fragment(config):
    lower = config.min_number_of_frames_per_fragment
    upper = config.max_number_of_frames_per_fragment
    mu = config.number_of_frames_per_fragment
    std = config.std_number_of_frames_per_fragment
    X = truncnorm((lower - mu) / std, (upper - mu) / std, loc=mu, scale=std)
    number_of_frames_per_fragment = int(X.rvs(1))
    return number_of_frames_per_fragment

def generate_list_of_blobs(portraits, centroids, config):
    blobs_in_video = []
    frames_in_fragment = 0
    number_of_fragments = 0

    print("\n***********Generating list of blobs")
    print("centroids shape ", centroids.shape)
    print("portraits shape", portraits.shape)
    for identity in range(config.number_of_animals):
        # print("*\n**identity ", identity)
        # decide length of first individual fragment for this identity
        number_of_frames_per_fragment = get_next_number_of_frames_in_fragment(config)
        frames_in_fragment = 0
        # print("number_of_frames_per_fragment, ", number_of_frames_per_fragment)
        # print("frames_in_fragment, ", frames_in_fragment)
        blobs_in_identity = []
        for frame_number in range(config.number_of_frames):
            # print("frame_number, ", frame_number)
            # print("identity, ", identity)
            centroid = centroids[frame_number,identity,:]
            image = portraits[frame_number,identity,:,:]

            blob = Blob(centroid, None, None, None,
                        number_of_animals = config.number_of_animals)
            blob.frame_number = frame_number
            blob._portrait = ((image - np.mean(image))/np.std(image)).astype("float16")
            blob._user_generated_identity = identity + 1

            if frame_number > 0 and frames_in_fragment <= number_of_frames_per_fragment + 2 and frames_in_fragment != 0:
                blob.previous = [blobs_in_identity[frame_number-1]]
                blobs_in_identity[frame_number-1].next = [blob]

            if frames_in_fragment <= number_of_frames_per_fragment:
                frames_in_fragment += 1
                # print("frames_in_fragment, ", frames_in_fragment)
            else:
                frames_in_fragment = 0
                number_of_fragments += 1
                number_of_frames_per_fragment = get_next_number_of_frames_in_fragment(config)
                # print("number_of_frames_per_fragment, ", number_of_frames_per_fragment)
                # print("frames_in_fragment, ", frames_in_fragment)

            blobs_in_identity.append(blob)

        blobs_in_video.append(blobs_in_identity)

    blobs_in_video = zip(*blobs_in_video)
    blobs_in_video = [list(blobs_in_frame) for blobs_in_frame in blobs_in_video]
    return blobs_in_video

class Dataset(object):
    def __init__(self, IMDB_codes = 'A', ids_codes = 'a', cluster = 0):
        self.IMDB_codes = IMDB_codes
        self.ids_codes = ids_codes
        self.cluster = cluster

        # Get list of IMDBPaths form IMDB_codes
        print('\nReading IMDB_codes and ids_codes...')
        if not int(self.cluster):
            self.datafolder = './'
        elif int(self.cluster):
            self.datafolder = '/admin/'
        self.IMDBsDict = {
                    'A': os.path.join(self.datafolder,'library','IMDBs','TU20160413_36dpf_60indiv_29938imperind_curvatureportrait_centroids_0.hdf5'),
                    'B': os.path.join(self.datafolder,'library','IMDBs','TU20160428_36dpf_60indiv_28010imperind_curvatureportrait_centroids_0.hdf5'),
                    'C': os.path.join(self.datafolder,'library','IMDBs','TU20160920_36dpf_64indiv_7731imperInd_curvatureportrait_centroids_0.hdf5'),
                    'D': os.path.join(self.datafolder,'library','IMDBs','TU20170131_31dpf_40indiv_34770imperind_curvatureportrait_centroids_0.hdf5'),
                    'E': os.path.join(self.datafolder,'library','IMDBs','TU20170201_31pdf_72indiv_38739ImPerInd_curvaturePortrait_centroids_0.hdf5'),
                    'F': os.path.join(self.datafolder,'library','IMDBs','TU20170202_31pdf_72indiv_38913imperind_curvatureportrait_centroids_0.hdf5'),
                    'G': os.path.join(self.datafolder,'library','IMDBs','tu20170131_31dpf_40indiv_34770imperind_fullbody_centroids_0.hdf5'),
                    'H': os.path.join(self.datafolder,'library','IMDBs','tu20170201_31pdf_72indiv_38739imperind_fullbody_centroids_0.hdf5'),
                    'I': os.path.join(self.datafolder,'library','IMDBs','tu20170202_31pdf_72indiv_38913imperind_fullbody_centroids_0.hdf5')
                    }
        self.IMDBPaths = []
        self.idsInIMDBs = []
        for (letter1,letter2) in zip(self.IMDB_codes,self.ids_codes):
            # print('\nletter1, ', letter1)
            self.IMDBPaths.append(self.IMDBsDict[letter1])
            IMDBName = getIMDBNameFromPath(self.IMDBsDict[letter1])
            # print('IMDBName, ', IMDBName)
            strain, age, numIndivIMDB, numImPerIndiv = getIMDBInfoFromName(IMDBName)
            # print('numIndivIMDB', numIndivIMDB)
            # print('letter2, ', letter2)
            if letter2 == 'a': # all ids
                ids = range(numIndivIMDB)
            elif letter2 == 'f': # first half idsInIMDBs
                ids = range(int(numIndivIMDB/2))
            elif letter2 == 's': # first half idsInIMDBs
                ids = range(int(numIndivIMDB/2),numIndivIMDB)
            # print('ids selected, ', ids)
            self.idsInIMDBs.append(ids)
        # print('IMDBPaths, ', self.IMDBPaths)
        # print('idsInIMDBs, ', self.idsInIMDBs)

    def loadIMDBs(self):
        # print('\n----------------------------------------------------------------')
        # print('Loading images and labels form the IMDBs selected')
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
            # print('\nExtracting imagaes from ', IMDBName)
            # print('The individuals selected from this IMDB are ',  idsInIMDB)
            # print('strain, ', strain)
            # print('age, ', age)
            # print('numIndivIMDB, ', numIndivIMDB)
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
            # print('images shape ', imagesIMDB.shape)
            # print('labels shape ', labelsIMDB.shape)
            # print('centroids shape ', centroidsIMDB.shape)
            self.images.append(imagesIMDB)
            self.labels.append(labelsIMDB)
            self.centroids.append(centroidsIMDB)
            # print('The labels added are, ', np.unique(labelsIMDB))
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
        # print('images shape ', self.images.shape)
        # print('labels shape ', self.labels.shape)
        # print('centroids shape ', self.centroids.shape)
        # print('labels ', np.unique(self.labels))
        self.minimum_number_of_images_per_animal = np.min([np.sum(self.labels == i) for i in np.unique(self.labels)])
        self.identities = np.unique(self.labels)
        # print('num images per label, ', self.minimum_number_of_images_per_animal)
        # print('----------------------------------------------------------------\n')

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
