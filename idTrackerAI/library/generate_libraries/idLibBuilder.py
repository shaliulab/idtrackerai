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
sys.path.append('./utils')
sys.path.append('./preprocessing')
sys.path.append('./library')
sys.path.append('./')

from segmentation_library import segment
# from fragmentation import *
from get_portraits import portrait
from video_utils_library import getSegmPaths, check_background_substraction
from py_utils import scanFolder, getExistentFiles_library, loadFile, saveFile, natural_sort, createFolder
from GUI_utils import selectDir, getInput, selectOptions, ROISelectorPreview_library, selectPreprocParams_library, playFragmentation_library
from library_utils import portraitsToIMDB, retrieveInfoLib, assignCenters

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    ''' ************************************************************************
    Selecting library directory
    ************************************************************************ '''
    initialDir = '/media/lab/idZebLib_TU20160413_34_36dpf/idZebLib/TU20160413/36dpf'
    libPath = selectDir(initialDir)

    ageInDpf, preprocessing, subDirs, strain = retrieveInfoLib(libPath, preprocessing = "curvaturePortrait")
    # subDirs = subDirs[:4]
    print 'ageInDpf, ', ageInDpf
    print 'strain, ', strain
    group = 0
    labelsIMDB = []
    portraitsIMDB = []
    bodiesIMDB = []
    bodyblobsIMDB = []
    centroidsIMDB = []
    nosesIMDB = []
    head_centroidsIMDB = []
    areasIMDB = []
    numImagesList = []
    numIndivIMDB = 0
    totalNumImages = 0
    setParams = getInput('Set preprocessing parameters','Do you want to set the parameters for the preprocessing of each video? ([y]/n)')
    runPreproc = getInput('Run preprocessing','Do you want to run the preprocessing? ([y]/n)')
    buildLib = getInput('Build library','Do you want to build the library? ([y]/n)')
    print subDirs
    for i, subDir in enumerate(subDirs):
        print '-----------------------'
        print 'preprocessing ', subDir

        ''' Path to video/s '''
        path = libPath + '/' + subDir
        extensions = ['.avi', '.mp4']
        videoPath = natural_sort([v for v in os.listdir(path) if isfile(path +'/'+ v) if any( ext in v for ext in extensions)])[0]
        videoPath = path + '/' + videoPath
        videoPaths = scanFolder(videoPath)
        createFolder(videoPath)
        frameIndices, segmPaths = getSegmPaths(videoPaths)

        ''' ************************************************************************
        Set preprocessing parameters
        ************************************************************************ '''
        if setParams == 'y' or setParams == '':
            setThisPrepParams = getInput('Set preprocessing parameters','Do you want to set parameters for this subDir ('+ subDir +')? ([y]/n)')
            print 'setThisPrepParams', setThisPrepParams
            if setThisPrepParams == 'n':
                continue
            elif setThisPrepParams == 'y' or setThisPrepParams == '':
                ''' ************************************************************************
                GUI to select the preprocessing parameters
                *************************************************************************'''
                prepOpts = selectOptions(['bkg', 'ROI'], None, text = 'Do you want to do BKG or select a ROI?  ')
                useBkg = prepOpts['bkg']
                useROI =  prepOpts['ROI']

                #Check for preexistent files generated during a previous session. If they
                #exist and one wants to keep them they will be loaded
                processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragments','portraits']
                existentFiles, srcSubFolder = getExistentFiles_library(videoPath, processesList, segmPaths)
                print 'List of processes finished, ', existentFiles
                print '\nSelecting files to load from previous session...'
                loadPreviousDict = selectOptions(processesList, existentFiles, text='Steps already processed in this video \n (check to load from ' + srcSubFolder + ')')

                usePreviousROI = loadPreviousDict['ROI']
                use_previous_background = loadPreviousDict['bkg']
                usePreviousPrecParams = loadPreviousDict['preprocparams']
                print 'usePreviousROI set to ', usePreviousROI
                print 'use_previous_background set to ', use_previous_background
                print 'usePreviousPrecParams set to ', usePreviousPrecParams

                ''' ROI selection/loading '''
                width, height, mask, centers = ROISelectorPreview_library(videoPaths, useROI, usePreviousROI, numSegment=0)
                ''' BKG computation/loading '''
                bkg = check_background_substraction(videoPaths, useBkg, use_previous_background, 0, width, height)

                ''' Selection/loading preprocessing parameters '''
                preprocParams = selectPreprocParams_library(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg, frameIndices)
                print 'The video will be preprocessed according to the following parameters: ', preprocParams

                cv2.namedWindow('Bars')

        ''' ************************************************************************
        Preprocessing
        ************************************************************************ '''
        if runPreproc == 'y' or runPreproc == '':
            processesList = ['ROI', 'bkg', 'preprocparams', 'segmentation','fragmentation','portraits']
            loadPreviousDict, srcSubFolder = getExistentFiles_library(videoPath, processesList, segmPaths)
            useBkg = int(loadPreviousDict['bkg'])
            useROI = loadPreviousDict['ROI']
            preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl='pkl')
            numAnimalsInGroup = preprocParams['numAnimals']

            ''' ************************************************************************
            Segmentation
            ************************************************************************ '''
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            centers = loadFile(videoPaths[0], 'centers')
            mask = np.asarray(loadFile(videoPaths[0], 'ROI'))
            if useBkg:
                bkg = loadFile(videoPaths[0], 'bkg', hdfpkl='pkl')
            else:
                bkg = None
            preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
            if not int(loadPreviousDict['segmentation']):
                EQ = 0
                print 'The preprocessing parameters dictionary loaded is ', preprocParams
                segment(videoPaths, preprocParams, mask, centers, useBkg, bkg, EQ)

            ''' ************************************************************************
            Fragmentation
            *************************************************************************'''
            ''' Group and number of individuals '''
            # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
            nameElements = subDir.split('_')
            if 'camera' in nameElements: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
                transform = 'rotation'
                video = int(nameElements[3])
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 3:
                transform = 'translation'
                video = int(nameElements[2])
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 2:
                transform = 'none'
                video = 1
                group += 1
                numIndivIMDB += numAnimalsInGroup

            print '************************************************************'
            print 'Video ', video
            print 'Transform, ', transform
            print 'Group, ', group
            print 'numAnimalsInGroup, ', numAnimalsInGroup
            print 'numIndivIMDB, ', numIndivIMDB
            print '************************************************************'

            if not int(loadPreviousDict['fragmentation']):
                print 'centers, ', centers
                centers = centers.drop_duplicates()
                centers = centers.reset_index(drop=True)
                print 'centers, ', centers
                if len(centers) != numAnimalsInGroup:
                    print 'centers, ', centers
                    print 'numAnimalsInGroup, ', numAnimalsInGroup
                    raise ValueError('The number of centers to assign identities is different to the number of animals in the group')
                dfGlobal = assignCenters(segmPaths,centers,video,transform = transform)
                print dfGlobal.columns
                playFragmentation_library(videoPaths,segmPaths,dfGlobal,visualize=False)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            ''' ************************************************************************
            Portraying
            ************************************************************************ '''
            if not int(loadPreviousDict['portraits']):
                videoInfo = loadFile(videoPaths[0], 'videoInfo')
                height = videoInfo['height']
                width = videoInfo['width']
                portrait(segmPaths, dfGlobal, height, width)
            # portraits = loadFile(videoPaths[0], 'portraits', time=0)

        if buildLib == 'y' or buildLib == '':
            ''' ************************************************************************
            Build images and labels array
            ************************************************************************ '''
            preprocParams= loadFile(videoPaths[0], 'preprocparams')
            # preprocParams = preprocParams.to_dict()[0]
            numAnimalsInGroup = preprocParams['numAnimals']
            print 'numAnimalsInGroup, ', numAnimalsInGroup
            ''' Group and number of individuals '''
            # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
            nameElements = subDir.split('_')
            if 'camera' in nameElements: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
                transform = 'rotation'
                video = int(nameElements[3])
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 3:
                transform = 'translation'
                video = int(nameElements[2])
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 2:
                transform = 'none'
                video = 1
                if i != 0:
                    group += 1
                numIndivIMDB += numAnimalsInGroup
            print 'Video, ', video
            print 'transform, ', transform
            print 'numIndivIMDB, ', numIndivIMDB
            print 'group, ', group

            portraits = loadFile(videoPaths[0], 'portraits')
            groupNum = i
            labels, portraits, bodies, bodyblobs, centroids, noses, head_centroids, areas = portraitsToIMDB(portraits, numAnimalsInGroup, group)
            print('portraits shape, ', portraits.shape)
            print('bodies shape, ', bodies.shape)
            print 'labels shape, ', labels.shape
            print 'labels, ', np.unique(labels)
            print 'images per indiv, ', [np.sum(labels==i) for i in np.unique(labels)]

            labelsIMDB.append(labels)
            portraitsIMDB.append(portraits)
            bodiesIMDB.append(bodies)
            bodyblobsIMDB.append(bodyblobs)
            centroidsIMDB.append(centroids)
            nosesIMDB.append(noses)
            head_centroidsIMDB.append(head_centroids)
            areasIMDB.append(areas)

            totalNumImages += labels.shape[0]

            numImagesList.append(labels.shape[0])

            print 'total number of images, ', numImagesList

    ''' ************************************************************************
    Save IMDB to hdf5
    ************************************************************************ '''
    if buildLib == 'y' or buildLib == '':
        preprocParams= loadFile(videoPaths[0], 'preprocparams')
        # preprocParams = preprocParams.to_dict()[0]
        numAnimalsInGroup = preprocParams['numAnimals']
        labelsIMDB = np.concatenate(labelsIMDB, axis = 0)
        portraitsIMDB = np.concatenate(portraitsIMDB, axis = 0)
        bodiesIMDB = np.concatenate(bodiesIMDB, axis = 0)
        bodyblobsIMDB = np.concatenate(bodyblobsIMDB, axis = 0)
        centroidsIMDB = np.concatenate(centroidsIMDB, axis = 0)
        nosesIMDB = np.concatenate(nosesIMDB, axis = 0)
        head_centroidsIMDB = np.concatenate(head_centroidsIMDB, axis = 0)
        areasIMDB = np.concatenate(areasIMDB, axis = 0)
        minimalNumImagesPerIndiv = int(np.min([np.sum(labelsIMDB == i) for i in np.unique(labelsIMDB)]))
        print '\n****************************************************************'
        print 'numIndivIMDB, ', numIndivIMDB
        print 'totalNumImages, ', totalNumImages
        print 'minNumImages, ', minimalNumImagesPerIndiv
        print 'labelsIMDB, ', np.unique(labelsIMDB)
        print 'num images per indiv, ', [np.sum(labelsIMDB == i) for i in np.unique(labelsIMDB)]

        nameDatabase =  strain + '_' + ageInDpf + '_' + str(numIndivIMDB) + 'indiv_' + str(int(minimalNumImagesPerIndiv)) + 'ImPerInd_' + 'bodyblobs'
        if not os.path.exists(libPath + '/IMDBs'): # Checkpoint folder does not exist
            os.makedirs(libPath + '/IMDBs') # we create a checkpoint folder
        else:
            if os.path.isfile(libPath + '/IMDBs/' + nameDatabase + '_0.hdf5'):
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

        f = h5py.File(libPath + '/IMDBs/' + nameDatabase + '_%i.hdf5', driver='family')
        grp = f.create_group("database")


        dset1 = grp.create_dataset("labels", labelsIMDB.shape, dtype='i')
        dset2 = grp.create_dataset("bodyblobs", bodyblobsIMDB.shape, dtype='f')
        dset3 = grp.create_dataset("centroids", centroidsIMDB.shape, dtype='f')
        dset4 = grp.create_dataset("noses", nosesIMDB.shape, dtype='f')
        dset5 = grp.create_dataset("head_centroids", head_centroidsIMDB.shape, dtype='f')
        dset6 = grp.create_dataset("areas", areasIMDB.shape, dtype='f')

        dset1[...] = labelsIMDB
        dset2[...] = bodyblobsIMDB
        #dset3[...] = bodiesIMDB
        #dset4[...] = bodyblobsIMDB
        dset3[...] = centroidsIMDB
        dset4[...] = nosesIMDB
        dset5[...] = head_centroidsIMDB
        dset6[...] = areasIMDB

        grp.attrs['originalMatPath'] = libPath
        grp.attrs['numIndiv'] = numIndivIMDB
        grp.attrs['strain'] = strain
        # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
        grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
        grp.attrs['ageInDpf'] = ageInDpf
        grp.attrs['preprocessing'] = preprocessing

        pprint([item for item in grp.attrs.iteritems()])

        f.close()

        print 'Database saved as %s ' % nameDatabase
