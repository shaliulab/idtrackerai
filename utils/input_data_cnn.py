import numpy as np
import h5py
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, one_hot_to_dense
import cPickle as pickle
import os
from pprint import *
from py_utils import *
import json
import msgpack as msg
import msgpack_numpy as msgnp

def matToH5(pathToDatabase):

    def retrieveInfo(pathToDatabase):
        nameFile = os.path.split(pathToDatabase)[-1]
        features = nameFile.split("_")

        # ageInDpf
        ageInDpf = [s for s in features if 'dpf' in s][0]

        # preprocessing
        preprocessing = [s for s in features if 'rot' in s or 'noBG' in s or 'pad' in s or 'reSize' in s ][0]

        return ageInDpf, preprocessing, nameFile

    ageInDpf, preprocessing, nameFile = retrieveInfo(pathToDatabase)

    Fish = {}
    with h5py.File(pathToDatabase, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'Matlab database %s loaded' % nameFile

    data = Fish['data'].astype(np.float32)
    # get image dimensions
    imsize = data[0].shape
    resolution = np.prod(imsize)
    # get labels
    label = Fish['labels'].astype(np.int32)
    label = np.squeeze(label)

    # compute number of individuals (classes)
    numIndiv = len(list(set(label)))
    numImagesPerIndiv =  np.true_divide(len(label),numIndiv)
    # get labels in {0,...,numIndiv-1}
    label = np.subtract(label,1)

    if not os.path.exists('../data'): # Checkpoint folder does not exist
        os.makedirs('../data') # we create a checkpoint folder

    nameDatabase =  ageInDpf + '_' + str(numIndiv) + 'indiv_' + str(int(numImagesPerIndiv)) + 'ImPerInd_' + preprocessing
    f = h5py.File('../data/' + nameDatabase + '_%i.hdf5', driver='family')
    grp = f.create_group("database")


    dset1 = grp.create_dataset("images", data.shape, dtype='f')
    dset2 = grp.create_dataset("labels", label.shape, dtype='i')

    dset1[...] = data
    dset2[...] = label

    grp.attrs['originalMatPath'] = pathToDatabase
    grp.attrs['numIndiv'] = numIndiv
    grp.attrs['imageSize'] = imsize
    grp.attrs['numImagesPerIndiv'] = numImagesPerIndiv
    grp.attrs['ageInDpf'] = ageInDpf
    grp.attrs['preprocessing'] = preprocessing

    pprint([item for item in grp.attrs.iteritems()])

    f.close()

    print 'Database saved as %s ' % nameDatabase

# matlabImdb = ['imdb_60indiv_11000_11000_32_rotateAndCrop_25dpf_s1_perm_0.mat', 'imdb_60indiv_11000_11000_32_rotateAndCrop_36dpf_s1_perm_0.mat']
# matlabImdb = ['imdb_15indiv_15000_1000_32_rotateAndCrop_25dpf_s1.mat']
# [matToH5('../matlabexports/' + m) for m in matlabImdb]

def loadDataBase(imdbTrain, numIndivTrainTest, numTrain, numTest, numRef=50, ckpt_folder="", imdbTest = None):
    '''
    imdbTrain: name of the database used for training and validation
    imdbTest: name of the databse used for testing
    numIndivTrainTest: number of individuals of imdbTrainVal used for training
    numTrain: number of images (per individual) for training (10% for validation)
    numTest: number of images (per individual) for testing
    '''

    def checkDatabase(imdb):
        # checks if the dataset given in input is already stored in the folder data
        if not os.path.exists('../data/' + imdb + '_0.hdf5'):
            raise ValueError('The database ' + imdb + ' does not exist in the directory /data. Copy there your database or create it (if .mat use matToH5)')

    def dimensionChecker(shape, dim):
        # compares two tuples (taking the order into account). Here it is used to
        # test the dimensionality of tensors (ndarrays).
        if shape != dim:
            raise ValueError('something is wrong! Expected dimension: ' + str(dim) + ' found: ' + str(shape) )

    def getVarAttrFromHdf5(database):
        # collect the info
        groups = database.keys()
        grp = database['database']
        datanames = grp.keys()
        images = grp['images'][()]
        labels = grp['labels'][()]
        # info = [item for item in grp.attrs.iteritems()]
        return grp, images, labels

    def getAttrsFromGroup(grp, variables):
        # retrieve an array from a h5py file
        return [grp.attrs[var] for var in variables]

    def permuter(N,name,load):
        # creates a permutation of N elements and stores it if load is False,
        # otherwise it loads it.
        print 'Creating permutation for %s' % name
        if not load:
            perm = np.random.permutation(N)
            # Save a permutation into a pickle file.
            permutation = { "perm": perm }
            pickle.dump( permutation, open( "../temp/permutation_" + name + ".pkl", "wb" ) )
            print ' No permutation exists, new one created'
        else:
            permutation = pickle.load( open( "../temp/permutation_" + name + ".pkl", "rb" ) )
            print ' Permutation loaded'
            perm = permutation['perm']

        return perm

    def sliceDatabase(images, labels, indicesIndiv):
        ''' Select images and labels relative to a subset of individuals'''
        print 'Slicing database...'
        images = np.array(flatten([images[labels==ind] for ind in indicesIndiv]))
        labels = np.array(flatten([i*np.ones(sum(labels==ind)).astype(int) for i,ind in enumerate(indicesIndiv)]))
        return images, labels

    def splitter(images, labels, numImages, numIndiv, imsize, numRef):
        # split (90-10%) a permuted database according to the number of requested images
        # remeber to PERMUTE images and labels before splitting them!
        numImages = numImages * numIndiv
        resolution = np.prod(imsize)
        num_val = int(np.ceil(np.true_divide(numImages,10)))
        num_train = int(numImages - num_val)

        X_train = images[:num_train]
        Y_train = labels[:num_train]
        X_val = images[num_train:num_train+num_val]
        Y_val = labels[num_train:num_train+num_val]

        # reshape images
        X_train = np.reshape(X_train, [num_train, resolution])
        X_val = np.reshape(X_val, [num_val, resolution])
        # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
        Y_train = dense_to_one_hot(Y_train, n_classes=numIndiv)
        Y_val = dense_to_one_hot(Y_val, n_classes=numIndiv)

        # # take references from validation
        # if numRef > 0:
        #     X_val, Y_val, X_ref, Y_ref = refTaker(X_val, Y_val, numRef, numIndiv)
        # else:
        #     X_ref = []
        #     Y_ref = []

        return X_train, Y_train, X_val, Y_val #, X_ref, Y_ref

    def cropper(images, labels, numImages, numIndiv, imsize, numRef,loadRefPerm):
        # crop a permuted database according to the number of requested images
        # remeber to PERMUTE images and labels before cropping them!
        resolution = np.prod(imsize)
        numImages = numImages * numIndiv
        X_test = images[:numImages]
        Y_test = labels[:numImages]

        # reshape images
        X_test = np.reshape(X_test, [numImages, resolution])
        # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
        Y_test = dense_to_one_hot(Y_test, n_classes=numIndiv)

        # take references from test
        if numRef > 0:
            X_test, Y_test, X_ref, Y_ref = refTaker(X_test, Y_test, numRef, numIndiv,loadRefPerm)
        else:
            print 'Zero references requested, the list of references will be empty...'
            X_ref = np.asarray([])
            Y_ref = np.asarray([])

        return X_test, Y_test, X_ref, Y_ref

    def refTaker(X,Y,numRef,numIndiv,loadRefPerm):
        X_ref = []
        Y_ref = []
        # Maybe improve this part (GPU?)
        for i in range(numIndiv):
            refIndices = np.where(one_hot_to_dense(Y)==i)[0][:numRef]

            X_ref.append(X[refIndices])
            Y_ref.append(Y[refIndices])
            X = np.delete(X,refIndices,axis=0)
            Y = np.delete(Y,refIndices,axis=0)

        # We need to permute the references since they have been created in order
        X_ref = np.asarray(flatten(X_ref))
        Y_ref = np.asarray(flatten(Y_ref))
        perm = permuter(len(Y_ref),'references',loadRefPerm)
        X_ref = X_ref[perm]
        Y_ref = Y_ref[perm]

        return X, Y, X_ref, Y_ref

    # check if the train database exists, and load it!
    checkDatabase(imdbTrain)
    with h5py.File("../data/" + imdbTrain + '_%i.hdf5', 'r', driver='family') as databaseTrain:
        [databaseTrainInfo, imagesTrain, labelsTrain] = getVarAttrFromHdf5(databaseTrain)
        [imsizeTrain,numIndivImdbTrain,numImagesPerIndivTrain] = getAttrsFromGroup(databaseTrainInfo,['imageSize', 'numIndiv','numImagesPerIndiv'])
        imsizeTrain = tuple(imsizeTrain)
        numImagesPerIndivTrain =  int(numImagesPerIndivTrain)
        print([item for item in databaseTrainInfo.attrs.iteritems()])

    print 'database %s loaded' %imdbTrain

    if numIndivImdbTrain < numIndivTrainTest:
        raise ValueError('The number of individuals for training (' + str(numIndivTrainTest) + ') cannot exceed the number of individuals in the database (' + str(numIndivImdbTrain) + ')')

    # Get images, labels, and indices of individuals for the train set
    permIndivTrain = permuter(numIndivImdbTrain,'individualsTrain',os.path.exists(ckpt_folder))
    indivTrain = permIndivTrain[:numIndivTrainTest]
    permImagesTrain = permuter(numImagesPerIndivTrain*numIndivTrainTest,'imagesTrain',os.path.exists(ckpt_folder))

    imagesTrainS, labelsTrainS = sliceDatabase(imagesTrain, labelsTrain, indivTrain)

    imagesTrainS = imagesTrainS[permImagesTrain]
    labelsTrainS = labelsTrainS[permImagesTrain]

    X_train, Y_train, X_val, Y_val = splitter(imagesTrainS, labelsTrainS, numTrain, numIndivTrainTest, imsizeTrain, numRef)

    # check train's dimensions
    cardTrain = int(np.ceil(np.true_divide(np.multiply(numTrain,9),10)))*numIndivTrainTest
    dimTrainL = (cardTrain, numIndivTrainTest)
    dimTrainI = (cardTrain, imagesTrain.shape[2]*imagesTrain.shape[3])

    dimensionChecker(X_train.shape, dimTrainI)
    dimensionChecker(Y_train.shape, dimTrainL)
    # check val's dimensions
    cardVal = int(np.ceil(np.true_divide(numTrain,10)))*numIndivTrainTest
    dimValL = (cardVal, numIndivTrainTest)
    dimValI = (cardVal, imagesTrain.shape[2] * imagesTrain.shape[3])

    dimensionChecker(X_val.shape, dimValI)
    dimensionChecker(Y_val.shape, dimValL)

    # check ref's dimensions
    # cardRef = numRef*numIndivTrainTest
    # dimRefL = (cardRef, numIndivTrainTest)
    # dimRefI = (cardRef, imagesTrain.shape[2] * imagesTrain.shape[3])
    #
    # dimensionChecker(X_ref.shape, dimRefI)
    # dimensionChecker(Y_ref.shape, dimRefL)
    # Get images, labels, and indices of individuals of the test set
    if numTest > 0:
        if imdbTest != None:
            print 'Loading the database for testing...'
            checkDatabase(imdbTest)
            with h5py.File("../data/" + imdbTest + '_%i.hdf5', 'r', driver='family') as databaseTest:
                [databaseTestInfo, imagesTest, labelsTest] = getVarAttrFromHdf5(databaseTest)
                [imsizeTest,numIndivImdbTest,numImagesPerIndivTest] = getAttrsFromGroup(databaseTestInfo,['imageSize', 'numIndiv','numImagesPerIndiv'])
                imsizeTest = tuple(imsizeTest)
                numImagesPerIndivTest = int(numImagesPerIndivTest)
                print([item for item in databaseTestInfo.attrs.iteritems()])

            print 'database %s loaded' %imdbTest

            if numIndivImdbTest < numIndivTrainTest:
                raise ValueError('The number of individuals for testing (' + str(numIndivTrainTest) + ') cannot exceed the number of individuals in the database (' + str(numIndivImdbTest) + ')')
            if imsizeTest != imsizeTrain:
                raise ValueError('The size of test images (' + str(imsizeTest) + ') is not compatible with the one of train images (' + str(imsizeTest) + ')')

            permIndivTest = permuter(numIndivImdbTest,'individualsTest',os.path.exists(ckpt_folder))
            indivTest = permIndivTest[:numIndivTrainTest]
        else:
            if numIndivImdbTrain < 2*numIndivTrainTest:
                raise ValueError('The total number of individuals required (' + str(numIndivTrainTest) + ') has to be less than (' + str(np.true_divide(numIndivImdbTrain,2)) + ')')

            imagesTest = imagesTrain
            labelsTest = labelsTrain
            permIndivTest = permIndivTrain
            indivTest = permIndivTrain[numIndivTrainTest:2*numIndivTrainTest]
            imsizeTest = imsizeTrain
            numImagesPerIndivTest = numImagesPerIndivTrain

        permImagesTest = permuter(numImagesPerIndivTest*numIndivTrainTest,'imagesTest',os.path.exists(ckpt_folder))
        imagesTestS, labelsTestS = sliceDatabase(imagesTest, labelsTest, indivTest)
        imagesTestS = imagesTestS[permImagesTest]
        labelsTestS = labelsTestS[permImagesTest]
        X_test, Y_test, X_ref, Y_ref = cropper(imagesTestS, labelsTestS, numTest, numIndivTrainTest, imsizeTest, numRef,os.path.exists(ckpt_folder))
        # check test's dimensions
        cardTest = (numTest - numRef) * numIndivTrainTest
        dimTestL = (cardTest, numIndivTrainTest)
        dimTestI = (cardTest, imagesTest.shape[2] * imagesTest.shape[3])

        dimensionChecker(X_test.shape, dimTestI)
        dimensionChecker(Y_test.shape, dimTestL)

    else:
        print 'The number of images for test is zero.'
        X_test = np.asarray([])
        Y_test = np.asarray([])
        X_ref = np.asarray([])
        Y_ref = np.asarray([])

    return numIndivTrainTest, imsizeTrain, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_ref, Y_ref

# loadDataBase(imdbTrain, numIndivTrainTest, numTrain, numTest, ckpt_folder, imdbTest = None):
# numIndivTrainTest, imsizeTrain, X_train, Y_train, X_val, Y_val, X_test, Y_test, X_ref, Y_ref = loadDataBase('25dpf_60indiv_22000ImPerInd_rotateAndCrop', 10, 1000, 100,10)

def dataHelper0(path, num_train, num_test, num_valid, ckpt_folder):
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    data = Fish['data'].astype(np.float32)
    # get image dimensions
    imsize = data[0].shape
    resolution = np.prod(imsize)
    # get labels
    label = Fish['labels'].astype(np.int32)
    label = np.squeeze(label)

    # compute number of individuals (classes)
    numIndiv = len(list(set(label)))
    # get labels in {0,...,numIndiv-1}
    label = np.subtract(label,1)

    print 'permuting the dataset'
    if not os.path.exists(ckpt_folder):
        perm = np.random.permutation(len(label))
        # Save a permutation into a pickle file.
        permutation = { "perm": perm }
        pickle.dump( permutation, open( "../temp/permutation.p", "wb" ) )
        print ' No permutation exists, new one created'
    else:
        permutation = pickle.load( open( "../temp/permutation.p", "rb" ) )
        print ' Permutation loaded'
        perm = permutation['perm']

    label = label[perm]
    data = data[perm]


    print 'splitting data in train, test and validation'
    N = num_train*numIndiv # of training data
    N_test = num_test*numIndiv # of test data
    N_val = num_valid*numIndiv # validation data

    X_train = data[:N]
    X_test = data[N:N+N_test]
    X_valid = data[N+N_test:N+N_test+N_val]
    y_train = label[:N]
    y_test = label[N:N+N_test]
    y_valid = label[N+N_test:N+N_test+N_val]

    # reshape images
    X_train = np.reshape(X_train, [N, resolution])
    X_test = np.reshape(X_test, [N_test, resolution])
    X_valid = np.reshape(X_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    Y_train = dense_to_one_hot(y_train, n_classes=numIndiv)
    Y_valid = dense_to_one_hot(y_valid, n_classes=numIndiv)
    Y_test = dense_to_one_hot(y_test, n_classes=numIndiv)

    return numIndiv, imsize, X_train, X_valid, X_test, Y_train, Y_valid, Y_test
