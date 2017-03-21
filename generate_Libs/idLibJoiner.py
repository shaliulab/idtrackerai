# Import standard libraries
import os
from os.path import isdir, isfile
import sys
import glob
import numpy as np
import cPickle as pickle

# Import third party libraries
import cv2
from pprint import pprint
import h5py

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')

# from segmentation import *
# from fragmentation import *
# from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
from library_utils import *
from input_data_cnn import *









if __name__ == '__main__':

    #Ask number of libraries to join
    numLibToJoin =  getInput('Number of libraries to join', 'How many libraries do you want to join?')
    try:
        numLibToJoin = int(numLibToJoin)
    except:
        raise valueError('Please the number of libraires to join has to be an integer number')

    #Ask libraries paths and identities
    libPaths = []
    libNames = []
    libIds = []
    libStrains = []
    libAges = []
    libNumIndiv = []
    for i in range(numLibToJoin):
        libPath = selectFile()
        IMDBName = getIMDBNameFromPath(libPath)
        strain, age, numIndiv, numImPerIndiv = getIMDBInfoFromName(IMDBName)

        print '****************************************************************'
        print 'libPath, ', libPath
        print 'IMDBName, ', IMDBName
        print 'strain, ', strain
        print 'age, ', age
        print 'numIndiv, ', numIndiv
        print 'numIMPerIndiv, ', numImPerIndiv

        #Ask identities range
        identities =  getInput('Identities', 'Please select the identities from this library (e.g. 0-5; 0,1,3,4, all)')
        if '-' in identities:
            idmin = int(identities.split('-')[0])
            idmax = int(identities.split('-')[1])
            identities = list(range(idmin,idmax))
        elif ',' in identities:
            identities = map(int,identities.split(','))
        elif identities == 'all':
            identities = range(numIndiv)
        print 'identities, ', identities

        libPaths.append(libPath)
        libNames.append(IMDBName)
        libIds.append(identities)
        libNumIndiv.append(len(identities))
        libStrains.append(strain)
        libAges.append(age)

    #Loop on the libraries
    AllImages = []
    AllLabels = []
    numIndivIMDB = 0
    print '\n************** Entering loop to build libraries ******************'
    for i, (IMBDPath,identities)  in enumerate(zip(libPaths,libIds)):
        print '****************************************************************'
        print 'libName, ', libNames[i]
        print 'identities, ', identities
        _, images, labels, _, _, _ = loadIMDB(IMBDPath)

        images, labels = sliceDatabase(images, labels, identities)
        if i>0:
            labels += numIndivIMDB

        print 'Current labels', labels
        AllImages.append(images)
        AllLabels.append(labels)
        numIndivIMDB += len(identities)

    imagesIMDB = np.vstack(AllImages)
    labelsIMDB = np.vstack(AllLabels)
    numImPerIndiv = [len(AllLabels == i) for i in np.unique(AllLabels)]
    minimalNumImagesPerIndiv = np.min(numImPerIndiv)

    #Save new IMDB
    libPath = os.path.dirname(libPaths[0])
    nameDatabase =  'test'
    if not os.path.exists(libPath):
        os.makedirs(libPath)
    else:
        if os.path.isfile(libPath + '/' + nameDatabase + '_0.hdf5'):
            text = 'A IMDB already exist with this name (' + nameDatabase + '). Do you want to create a new one with a different name?'
            newName = getInput('Confirm selection','The IMDB already exist. Do you want to create a new one with a different name [y/n]?')
            if newName == 'y':
                nameDatabase = getInput('Insert new name: ','The current name is "' + nameDatabase + '"')
            elif newName == 'n':
                displayMessage('Overwriting IMDB','You are going to overwrite the current IMDB (' + nameDatabase + ').')
                for filename in glob.glob(libPath + '/IMDBs/' + nameDatabase + '_*.hdf5'):
                    os.remove(filename)
            else:
                raise ValueError('Invalid string, it must be "y" or "n"')


    f = h5py.File(libPath + '/' + nameDatabase + '_%i.hdf5', driver='family')
    grp = f.create_group("database")

    dset1 = grp.create_dataset("images", imagesIMDB.shape, dtype='f')
    dset2 = grp.create_dataset("labels", labelsIMDB.shape, dtype='i')

    dset1[...] = imagesIMDB
    dset2[...] = labelsIMDB

    grp.attrs['originalPath'] = libPaths
    grp.attrs['numIndiv'] = numIndivIMDB
    grp.attrs['imageSize'] = imsize
    grp.attrs['strain'] = libStrains
    # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
    grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
    grp.attrs['ageInDpf'] = libAges
    grp.attrs['preprocessing'] = preprocessing

    pprint([item for item in grp.attrs.iteritems()])

    f.close()

    print 'Database saved as %s ' % nameDatabase










    # cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...
    #
    # ''' ************************************************************************
    # Selecting library directory
    # ************************************************************************ '''
    # initialDir = '/media/lab/idZebLib_TU20160413_34_36dpf/idZebLib/TU20160413/36dpf'
    # libPath = selectDir(initialDir)
    #
    # ageInDpf, preprocessing, subDirs, strain = retrieveInfoLib(libPath, preprocessing = "curvaturePortrait")
    # print 'ageInDpf, ', ageInDpf
    # print 'strain, ', strain
    # group = 0
    # imagesIMDB = []
    # labelsIMDB = []
    # numImagesList = []
    # numIndivIMDB = 0
    # totalNumImages = 0
    # setParams = getInput('Set preprocessing parameters','Do you want to set the parameters for the preprocessing of each video? ([y]/n)')
    # runPreproc = getInput('Run preprocessing','Do you want to run the preprocessing? ([y]/n)')
    # buildLib = getInput('Build library','Do you want to build the library? ([y]/n)')
    # print subDirs
    # for i, subDir in enumerate(subDirs):
    #     print '-----------------------'
    #     print 'preprocessing ', subDir
    #
    #     ''' Path to video/s '''
    #     path = libPath + '/' + subDir
    #     extensions = ['.avi', '.mp4']
    #     videoPath = natural_sort([v for v in os.listdir(path) if isfile(path +'/'+ v) if any( ext in v for ext in extensions)])[0]
    #     videoPath = path + '/' + videoPath
    #     videoPaths = scanFolder(videoPath)
    #     createFolder(videoPath)
    #     frameIndices, segmPaths = getSegmPaths(videoPaths)
    #
    #     ''' ************************************************************************
    #     Set preprocessing parameters
    #     ************************************************************************ '''
    #     if setParams == 'y' or setParams == '':
    #         setThisPrepParams = getInput('Set preprocessing parameters','Do you want to set parameters for this subDir ('+ subDir +')? ([y]/n)')
    #         print 'setThisPrepParams', setThisPrepParams
    #         if setThisPrepParams == 'n':
    #             continue
    #         elif setThisPrepParams == 'y' or setThisPrepParams == '':
    #             ''' ************************************************************************
    #             GUI to select the preprocessing parameters
    #             *************************************************************************'''
    #             prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
    #             useBkg = prepOpts['bkg']
    #             useROI =  prepOpts['ROI']
    #
    #             #Check for preexistent files generated during a previous session. If they
    #             #exist and one wants to keep them they will be loaded
    #             processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragments','portraits']
    #             existentFiles, srcSubFolder = getExistentFiles(videoPath, processesList, segmPaths)
    #             print 'List of processes finished, ', existentFiles
    #             print '\nSelecting files to load from previous session...'
    #             loadPreviousDict = selectOptions(processesList, existentFiles, text='Steps already processed in this video \n (check to load from ' + srcSubFolder + ')')
    #
    #             usePreviousROI = loadPreviousDict['ROI']
    #             usePreviousBkg = loadPreviousDict['bkg']
    #             usePreviousPrecParams = loadPreviousDict['preprocparams']
    #             print 'usePreviousROI set to ', usePreviousROI
    #             print 'usePreviousBkg set to ', usePreviousBkg
    #             print 'usePreviousPrecParams set to ', usePreviousPrecParams
    #
    #             ''' ROI selection/loading '''
    #             width, height, mask, centers = ROISelectorPreview(videoPaths, useROI, usePreviousROI, numSegment=0)
    #             ''' BKG computation/loading '''
    #             bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, 0, width, height)
    #
    #             ''' Selection/loading preprocessing parameters '''
    #             preprocParams = selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg, frameIndices)
    #             print 'The video will be preprocessed according to the following parameters: ', preprocParams
    #
    #             cv2.namedWindow('Bars')
    #
    #     ''' ************************************************************************
    #     Preprocessing
    #     ************************************************************************ '''
    #     if runPreproc == 'y' or runPreproc == '':
    #         processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragmentation','portraits']
    #         loadPreviousDict, srcSubFolder = getExistentFiles(videoPath, processesList, segmPaths)
    #         useBkg = int(loadPreviousDict['bkg'])
    #         useROI = loadPreviousDict['ROI']
    #         preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl='pkl')
    #         numAnimalsInGroup = preprocParams['numAnimals']
    #
    #         ''' ************************************************************************
    #         Segmentation
    #         ************************************************************************ '''
    #         cv2.waitKey(1)
    #         cv2.destroyAllWindows()
    #         cv2.waitKey(1)
    #         centers = loadFile(videoPaths[0], 'centers')
    #         mask = np.asarray(loadFile(videoPaths[0], 'ROI'))
    #         if useBkg:
    #             bkg = loadFile(videoPaths[0], 'bkg', hdfpkl='pkl')
    #         else:
    #             bkg = None
    #         preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
    #         if not int(loadPreviousDict['segmentation']):
    #             EQ = 0
    #             print 'The preprocessing parameters dictionary loaded is ', preprocParams
    #             segment(videoPaths, preprocParams, mask, centers, useBkg, bkg, EQ)
    #
    #         ''' ************************************************************************
    #         Fragmentation
    #         *************************************************************************'''
    #         ''' Group and number of individuals '''
    #         # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
    #         nameElements = subDir.split('_')
    #         if 'camera' in nameElements: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
    #             transform = 'rotation'
    #             video = int(nameElements[3])
    #             if video == 1 and i != 0:
    #                 group += 1
    #             if video == 2:
    #                 numIndivIMDB += numAnimalsInGroup
    #         elif len(nameElements) == 3:
    #             transform = 'translation'
    #             video = int(nameElements[2])
    #             if video == 1 and i != 0:
    #                 group += 1
    #             if video == 2:
    #                 numIndivIMDB += numAnimalsInGroup
    #         elif len(nameElements) == 2:
    #             transform = 'none'
    #             video = 1
    #             group += 1
    #             numIndivIMDB += numAnimalsInGroup
    #
    #         print '************************************************************'
    #         print 'Video ', video
    #         print 'Transform, ', transform
    #         print 'Group, ', group
    #         print 'numAnimalsInGroup, ', numAnimalsInGroup
    #         print 'numIndivIMDB, ', numIndivIMDB
    #         print '************************************************************'
    #
    #         if not int(loadPreviousDict['fragmentation']):
    #             dfGlobal = assignCenters(segmPaths,centers,video,transform = transform)
    #             print dfGlobal.columns
    #             playFragmentation(videoPaths,segmPaths,dfGlobal,visualize=False)
    #             cv2.waitKey(1)
    #             cv2.destroyAllWindows()
    #             cv2.waitKey(1)
    #
    #         ''' ************************************************************************
    #         Portraying
    #         ************************************************************************ '''
    #         if not int(loadPreviousDict['portraits']):
    #             portrait(segmPaths, dfGlobal)
    #         # portraits = loadFile(videoPaths[0], 'portraits', time=0)
    #
    #     if buildLib == 'y' or buildLib == '':
    #         ''' ************************************************************************
    #         Build images and labels array
    #         ************************************************************************ '''
    #         preprocParams= loadFile(videoPaths[0], 'preprocparams')
    #         # preprocParams = preprocParams.to_dict()[0]
    #         numAnimalsInGroup = preprocParams['numAnimals']
    #         print 'numAnimalsInGroup, ', numAnimalsInGroup
    #         ''' Group and number of individuals '''
    #         # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
    #         nameElements = subDir.split('_')
    #         if 'camera' in nameElements: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
    #             transform = 'rotation'
    #             video = int(nameElements[3])
    #             if video == 1 and i != 0:
    #                 group += 1
    #             if video == 2:
    #                 numIndivIMDB += numAnimalsInGroup
    #         elif len(nameElements) == 3:
    #             transform = 'translation'
    #             video = int(nameElements[2])
    #             if video == 1 and i != 0:
    #                 group += 1
    #             if video == 2:
    #                 numIndivIMDB += numAnimalsInGroup
    #         elif len(nameElements) == 2:
    #             transform = 'none'
    #             video = 1
    #             group += 1
    #             numIndivIMDB += numAnimalsInGroup
    #         print 'Video, ', video
    #         print 'transform, ', transform
    #         print 'numIndivIMDB, ', numIndivIMDB
    #         print 'group, ', group
    #
    #         portraits = loadFile(videoPaths[0], 'portraits')
    #         groupNum = i
    #         imsize, images, labels = portraitsToIMDB(portraits, numAnimalsInGroup, group)
    #         print 'images, shape, ', images.shape
    #         print 'labels, shape, ', labels.shape
    #         imagesIMDB.append(images)
    #         labelsIMDB.append(labels)
    #
    #         totalNumImages += labels.shape[0]
    #
    #         numImagesList.append(labels.shape[0])
    #
    #         print numImagesList
    #
    # ''' ************************************************************************
    # Save IMDB to hdf5
    # ************************************************************************ '''
    # if buildLib == 'y' or buildLib == '':
    #     preprocParams= loadFile(videoPaths[0], 'preprocparams')
    #     # preprocParams = preprocParams.to_dict()[0]
    #     numAnimalsInGroup = preprocParams['numAnimals']
    #     averageNumImagesPerIndiv = int(np.divide(totalNumImages,numIndivIMDB))
    #     minimalNumImagesPerIndiv = int(np.divide(np.min(numImagesList),numAnimalsInGroup))*2
    #     print 'numIndivIMDB, ', numIndivIMDB
    #     print 'totalNumImages, ', totalNumImages
    #     print 'avNumImages, ', averageNumImagesPerIndiv
    #     print 'minNumImages, ', minimalNumImagesPerIndiv
    #     imagesIMDB = np.vstack(imagesIMDB)
    #     labelsIMDB = np.vstack(labelsIMDB)
    #
    #     ### TODO the dataset name should include the strain of the animals used
    #     nameDatabase =  strain + '_' + ageInDpf + '_' + str(numIndivIMDB) + 'indiv_' + str(int(minimalNumImagesPerIndiv)) + 'ImPerInd_' + preprocessing
    #     if not os.path.exists(libPath + '/IMDBs'): # Checkpoint folder does not exist
    #         os.makedirs(libPath + '/IMDBs') # we create a checkpoint folder
    #     else:
    #         if os.path.isfile(libPath + '/IMDBs/' + nameDatabase + '_0.hdf5'):
    #             text = 'A IMDB already exist with this name (' + nameDatabase + '). Do you want to create a new one with a different name?'
    #             newName = getInput('Confirm selection','The IMDB already exist. Do you want to create a new one with a different name [y/n]?')
    #             if newName == 'y':
    #                 nameDatabase = getInput('Insert new name: ','The current name is "' + nameDatabase + '"')
    #             elif newName == 'n':
    #                 displayMessage('Overwriting IMDB','You are going to overwrite the current IMDB (' + nameDatabase + ').')
    #                 for filename in glob.glob(libPath + '/IMDBs/' + nameDatabase + '_*.hdf5'):
    #                     os.remove(filename)
    #             else:
    #                 raise ValueError('Invalid string, it must be "y" or "n"')
    #
    #     f = h5py.File(libPath + '/IMDBs/' + nameDatabase + '_%i.hdf5', driver='family')
    #     grp = f.create_group("database")
    #
    #
    #     dset1 = grp.create_dataset("images", imagesIMDB.shape, dtype='f')
    #     dset2 = grp.create_dataset("labels", labelsIMDB.shape, dtype='i')
    #
    #     dset1[...] = imagesIMDB
    #     dset2[...] = labelsIMDB
    #
    #     grp.attrs['originalMatPath'] = libPath
    #     grp.attrs['numIndiv'] = numIndivIMDB
    #     grp.attrs['imageSize'] = imsize
    #     grp.attrs['strain'] = strain
    #     # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
    #     grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
    #     grp.attrs['ageInDpf'] = ageInDpf
    #     grp.attrs['preprocessing'] = preprocessing
    #
    #     pprint([item for item in grp.attrs.iteritems()])
    #
    #     f.close()
    #
    #     print 'Database saved as %s ' % nameDatabase
