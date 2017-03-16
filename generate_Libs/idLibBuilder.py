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

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')
sys.path.append('IdTrackerDeep/preprocessing')

from segmentation import *
from fragmentation import *
from get_portraits import *
from video_utils import *
from py_utils import *
from GUI_utils import *
from library_utils import *

if __name__ == '__main__':
    cv2.namedWindow('Bars') #FIXME If we do not create the "Bars" window here we have the "Bad window error"...

    ''' ************************************************************************
    Selecting library directory
    ************************************************************************ '''
    initialDir = '/media/lab/idZebLib_TU20160413_34_36dpf/idZebLib/TU20160413/36dpf'
    libPath = selectDir(initialDir)

    ageInDpf, preprocessing, subDirs = retrieveInfoLib(libPath, preprocessing = "curvaturePortrait")
    group = 0
    imagesIMDB = []
    labelsIMDB = []
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
                existentFiles, srcSubFolder = getExistentFiles(videoPath, processesList, segmPaths)
                print 'List of processes finished, ', existentFiles
                print '\nSelecting files to load from previous session...'
                loadPreviousDict = selectOptions(processesList, existentFiles, text='Steps already processed in this video \n (check to load from ' + srcSubFolder + ')')

                usePreviousROI = loadPreviousDict['ROI']
                usePreviousBkg = loadPreviousDict['bkg']
                usePreviousPrecParams = loadPreviousDict['preprocparams']
                print 'usePreviousROI set to ', usePreviousROI
                print 'usePreviousBkg set to ', usePreviousBkg
                print 'usePreviousPrecParams set to ', usePreviousPrecParams

                ''' ROI selection/loading '''
                width, height, mask, centers = ROISelectorPreview(videoPaths, useROI, usePreviousROI, numSegment=0)
                ''' BKG computation/loading '''
                bkg = checkBkg(videoPaths, useBkg, usePreviousBkg, 0, width, height)

                ''' Selection/loading preprocessing parameters '''
                preprocParams = selectPreprocParams(videoPaths, usePreviousPrecParams, width, height, bkg, mask, useBkg)
                print 'The video will be preprocessed according to the following parameters: ', preprocParams

        ''' ************************************************************************
        Preprocessing
        ************************************************************************ '''
        if runPreproc == 'y' or runPreproc == '':
            ''' ************************************************************************
            Segmentation
            ************************************************************************ '''
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            if not loadPreviousDict['segmentation']:
                preprocParams= loadFile(videoPaths[0], 'preprocparams',hdfpkl = 'pkl')
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
                affineTransform = 'rotation'
                video = nameElements[3]
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 3:
                affineTransform = 'translation'
                video = nameElements[2]
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 2:
                affineTransform = 'none'
                video = 1
                group += 1
                numIndivIMDB += numAnimalsInGroup
            print 'Video ', video

            if not loadPreviousDict['fragmentation']:
                assignCenters(videoPaths,centers,video,affineTransform = affineTransform)
                playFragmentation(videoPaths,False) # last parameter is to visualize or not
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            ''' ************************************************************************
            Portraying
            ************************************************************************ '''
            if not loadPreviousDict['portraits']:
                portrait(videoPaths)
            # portraits = loadFile(videoPaths[0], 'portraits', time=0)

        if buildLib == 'y' or buildLib == '':
            ''' ************************************************************************
            Build images and labels array
            ************************************************************************ '''
            preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
            preprocParams = preprocParams.to_dict()[0]
            numAnimalsInGroup = preprocParams['numAnimals']
            ''' Group and number of individuals '''
            # We assign the group number in a iterative way so that if one group is missing the labels of the IMDB are still iterative
            nameElements = subDir.split('_')
            if 'camera' in nameElements: ###NOTE this only works if the subDirs list alternates between camera 1 and 2 every time
                affineTransform = 'rotation'
                video = nameElements[3]
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 3:
                affineTransform = 'translation'
                video = nameElements[2]
                if video == 1 and i != 0:
                    group += 1
                if video == 2:
                    numIndivIMDB += numAnimalsInGroup
            elif len(nameElements) == 2:
                affineTransform = 'none'
                video = 1
                group += 1
                numIndivIMDB += numAnimalsInGroup
            print 'Video ', video
            
            portraits = loadFile(videoPaths[0], 'portraits', time=0)
            groupNum = i
            imsize, images, labels = portraitsToIMDB(portraits, numAnimalsInGroup, group)
            print 'images, shape, ', images.shape
            print 'labels, shape, ', labels.shape
            imagesIMDB.append(images)
            labelsIMDB.append(labels)

            totalNumImages += labels.shape[0]

            numImagesList.append(labels.shape[0])
            print numImagesList

    ''' ************************************************************************
    Save IMDB to hdf5
    ************************************************************************ '''
    if buildLib == 'y' or buildLib == '':
        preprocParams= loadFile(videoPaths[0], 'preprocparams',0)
        preprocParams = preprocParams.to_dict()[0]
        numAnimalsInGroup = preprocParams['numAnimals']

        averageNumImagesPerIndiv = int(np.divide(totalNumImages,numIndivIMDB))
        minimalNumImagesPerIndiv = int(np.divide(np.min(numImagesList),numAnimalsInGroup))*2
        imagesIMDB = np.vstack(imagesIMDB)
        labelsIMDB = np.vstack(labelsIMDB)

        ### TODO the dataset name should include the strain of the animals used
        nameDatabase =  ageInDpf + '_' + str(numIndivIMDB) + 'indiv_' + str(int(minimalNumImagesPerIndiv)) + 'ImPerInd_' + preprocessing
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


        dset1 = grp.create_dataset("images", imagesIMDB.shape, dtype='f')
        dset2 = grp.create_dataset("labels", labelsIMDB.shape, dtype='i')

        dset1[...] = imagesIMDB
        dset2[...] = labelsIMDB

        grp.attrs['originalMatPath'] = libPath
        grp.attrs['numIndiv'] = numIndivIMDB
        grp.attrs['imageSize'] = imsize
        # grp.attrs['averageNumImagesPerIndiv'] = averageNumImagesPerIndiv
        grp.attrs['numImagesPerIndiv'] = minimalNumImagesPerIndiv
        grp.attrs['ageInDpf'] = ageInDpf
        grp.attrs['preprocessing'] = preprocessing

        pprint([item for item in grp.attrs.iteritems()])

        f.close()

        print 'Database saved as %s ' % nameDatabase
