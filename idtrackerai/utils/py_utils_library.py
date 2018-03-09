# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals with correct identities in large
# animal collectives (submitted)

from __future__ import division
from itertools import groupby
import os
import glob
import re
import datetime
import pandas as pd
import numpy as np
import shutil
import cPickle as pickle
import sys
from pprint import pprint
import matplotlib
import logging
import matplotlib
import subprocess
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.py_utils")

### Git utils ###
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])

### Object utils ###
def append_values_to_lists(values, list_of_lists):
    list_of_lists_updated = []

    for l, value in zip(list_of_lists, values):
        l.append(value)
        list_of_lists_updated.append(l)

    return list_of_lists_updated

def set_attributes_of_object_to_value(object_to_modify, attributes_list, value = None):
    [setattr(object_to_modify, attribute, value) for attribute in attributes_list if hasattr(object_to_modify, attribute)]

def delete_attributes_from_object(object_to_modify, list_of_attributes):
    [delattr(object_to_modify, attribute) for attribute in list_of_attributes if hasattr(object_to_modify, attribute)]

### Dict utils ###
def getVarFromDict(dictVar,variableNames):
    ''' get variables from a standard python dictionary '''
    return [dictVar[v] for v in variableNames]

def maskArray(im1,im2,w1,w2):
    return np.add(np.multiply(im1,w1),np.multiply(im2,w2))

def flatten(l):
    ''' flatten a list of lists '''
    try:
        ans = [inner for outer in l for inner in outer]
    except:
        ans = [y for x in l for y in (x if isinstance(x, tuple) else (x,))]
    return ans

def cycle(l):
    ''' shift the list one element towards the right
    [a,b,c] -> [c,a,b] '''
    l.insert(0,l.pop())
    return l

def Ncycle(l,n):
    for i in range(n):
        l = cycle(l)
    return l

def countRate(array):
    # count repetitions of each element in array and returns the multiset
    # (el, multiplicity(el))
    # [1,1,2,1] outputs [(1,2), (2,1), (1,1)]
    return [(key,len(list(group))) for key, group in groupby(array)]

def countRateSet(array):
    # count repetitions of each element in array, by summing the multiplicity of
    # identical components
    # [1,1,2,1] outputs [(1,3), (2,1)]
    uniqueEl = list(set(array))
    ratePerElement = [(key,len(list(group))) for key, group in groupby(array)]
    return [(el,sum([pair[1] for pair in ratePerElement if pair[0]== el])) for el in uniqueEl]

def groupByCustom(array, keys, ind): #FIXME it can be done matricially
    """
    given an array to group and an array of keys returns the array grouped in a
    dictionary according to the keys listed at the index ind
    """
    dictionary = {i: [] for i in keys}
    for el in array:
        dictionary[el[ind]].append(el)

    return dictionary

def deleteDuplicates(array):
    # deletes duplicate sublists in list
    newArray = []
    delInds = []
    for i,elem in enumerate(array):
        if elem not in newArray:
            newArray.append(elem)
        else:
            delInds.append(i)

    return newArray,delInds


def ssplit2(seq,splitters):
    """
    split a list at splitters, if the splitted sequence is longer than 1
    """
    seq=list(seq)
    if splitters and seq:
        splitters=set(splitters).intersection(seq)
        if splitters:
            result=[]
            begin=0
            for end in range(len(seq)):
                if seq[end] in splitters:
                    if (end > begin and len(seq[begin:end])>1) :
                        result.append(seq[begin:end])
                    begin=end+1
            if begin<len(seq):
                result.append(seq[begin:])
            return result
    return [seq]

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def getSubfolders(folder):
    ''' returns subfolders of a given path'''
    return [os.path.join(folder, path) for path in os.listdir(folder) if os.path.isdir(os.path.join(folder, path))]

def getFiles(folder):
    ''' returns files of a given path'''
    return [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]

def getFilesAndSubfolders(folder):
    ''' returns files and subfodlers of a given path in two different lists'''
    files = [name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))]
    subfolders = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    return files, subfolders

def scanFolder(path):
    ### NOTE if the video selected does not finish with '_1' the scanFolder function won't select all of them. This can be improved
    paths = [path]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == '_1':
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

def get_spaced_colors_util(n, norm = False, black = True, cmap = 'jet'):
    RGB_tuples = matplotlib.cm.get_cmap(cmap)
    if norm:
        colors = [RGB_tuples(i / n) for i in range(n)]
    else:
        RGB_array = np.asarray([RGB_tuples(i / n) for i in range(n)])
        BRG_array = np.zeros(RGB_array.shape)
        BRG_array[:,0] = RGB_array[:,2]
        BRG_array[:,1] = RGB_array[:,1]
        BRG_array[:,2] = RGB_array[:,0]
        colors = [tuple(BRG_array[i,:] * 256) for i in range(n)]
    if black:
        black = (0., 0., 0.)
        colors.insert(0, black)
    return colors

def saveFile(path, variabletoSave, name, hdfpkl = 'hdf',sessionPath = '', nSegment = None):
    import cPickle as pickle
    """
    All the input are strings!!!
    path: path to the first segment of the video
    name: string to add to the name of the video (wihtout timestamps)
    folder: path to the folder in which the file has to be stored
    """
    # if os.path.exists(path)==False:
    #     raise ValueError("the video %s does not exist!" %path)
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    filename, extension = os.path.splitext(video)

    if name == 'segment' or name == 'segmentation':
        subfolder = '/preprocessing/segmentation/'
        if nSegment == None:
            nSegment = filename.split('_')[-1]# and before the number of the segment
        if hdfpkl == 'hdf':
            filename = 'segm_' + nSegment + '.pkl'
            pathToSave = folder + subfolder + filename
            variabletoSave.to_pickle(pathToSave)
        elif hdfpkl == 'pkl':
            filename = 'segm_' + nSegment + '.pkl'
            pathToSave = folder + subfolder+ filename
            pickle.dump(variabletoSave,open(pathToSave,'wb'))
    elif name == 'trajectories':
        filename = 'trajectories.pkl'
        pathToSave = sessionPath + '/' + filename
        pickle.dump(variabletoSave,open(pathToSave,'wb'))
    else:
        subfolder = '/preprocessing/'
        if hdfpkl == 'hdf':
            filename = name + '.pkl'
            if isinstance(variabletoSave, dict):
                variabletoSave = pd.DataFrame.from_dict(variabletoSave,orient='index')
            elif not isinstance(variabletoSave, pd.DataFrame):
                variabletoSave = pd.DataFrame(variabletoSave)
            pathToSave = folder + subfolder + filename
            variabletoSave.to_pickle(pathToSave)
        elif hdfpkl == 'pkl':
            filename = name + '.pkl'
            # filename = os.path.relpath(filename)
            pathToSave = folder + subfolder + filename
            pickle.dump(variabletoSave,open(pathToSave,'wb'))

    print 'You just saved ', pathToSave

def loadFile(path, name, hdfpkl = 'hdf',sessionPath = ''):
    """
    loads a pickle. path is the path of the video, while name is a string in the
    set {}
    """
    video = os.path.basename(path)
    folder = os.path.dirname(path)
    filename, extension = os.path.splitext(video)
    subfolder = ''

    if name  == 'segmentation':
        subfolder = '/preprocessing/segmentation/'
        nSegment = filename.split('_')[-1]
        if hdfpkl == 'hdf':
            filename = 'segm_' + nSegment + '.pkl'
            return pd.read_pickle(folder + subfolder + filename ), nSegment
        elif hdfpkl == 'pkl':
            filename = 'segm_' + nSegment + '.pkl'
            return pickle.load(open(folder + subfolder + filename) ,'rb'), nSegmen
    elif name == 'statistics':
        filename = 'statistics.pkl'
        return pickle.load(open(sessionPath + '/' + filename,'rb') )
    elif name == 'trajectories':
        filename = 'trajectories.pkl'
        return pickle.load(open(sessionPath + '/' + filename,'rb') )
    else:
        subfolder = '/preprocessing/'
        if hdfpkl == 'hdf':
            filename = name + '.pkl'
            return pd.read_pickle(folder + subfolder + filename )
        elif hdfpkl == 'pkl':
            filename = name + '.pkl'
            return pickle.load(open(folder + subfolder + filename,'rb') )

    print 'You just loaded ', folder + subfolder + filename

def check_and_change_video_path(video,old_video):
    current_video_folder = os.path.split(video.video_path)[0]
    old_video_folder = os.path.split(old_video.video_path)[0]
    old_video_session_name = old_video.session_folder
    if current_video_folder != old_video_folder:
        attributes_to_modify = {key: getattr(old_video, key) for key in old_video.__dict__
        if isinstance(getattr(old_video, key), basestring)
        and old_video_folder in getattr(old_video, key) }

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(old_video_folder, current_video_folder)
            setattr(old_video, key, new_value)

        if old_video.paths_to_video_segments is not None and len(old_video.paths_to_video_segments) != 0:
            new_paths_to_video_segments = []

            for path in old_video.paths_to_video_segments:
                new_paths_to_video_segments.append(path.replace(old_video_folder, current_video_folder))
            old_video._paths_to_video_segments = new_paths_to_video_segments

        ### update checkpoint files
        current_video_session_name = old_video.session_folder
        folders_to_check = ['_crossings_detector_folder',
                            '_pretraining_folder',
                            '_accumulation_folder']
        for folder in folders_to_check:
            if hasattr(old_video, folder) and getattr(old_video, folder) is not None:
                if folder == 'crossings_detector_folder':
                    checkpoint_path = os.path.join(old_video.crossings_detector_folder, 'checkpoint')
                    print(checkpoint_path)
                    if os.path.isfile(checkpoint_path):
                        old_video.update_tensorflow_checkpoints_file(checkpoint_path, old_video_session_name, current_video_session_name)
                    else:
                        logger.warn('No checkpoint found in %s ' %folder)
                else:
                    for sub_folder in ['conv', 'softmax']:
                        checkpoint_path = os.path.join(getattr(old_video,folder), sub_folder, 'checkpoint')
                        if os.path.isfile(checkpoint_path):
                            old_video.update_tensorflow_checkpoints_file(checkpoint_path, old_video_session_name, current_video_session_name)
                        else:
                            logger.warn('No checkpoint found in %s ' %os.path.join(getattr(old_video, folder), sub_folder))
    return old_video

def set_load_previous_dict(old_video, processes, existentFile):
    attributes = [ 'has_been_preprocessed',
                    'first_accumulation_finished',
                    'has_been_pretrained', 'second_accumulation_finished',
                    'has_been_assigned', 'has_duplications_solved',
                    'has_crossings_solved', 'has_trajectories',
                    'has_trajectories_wo_gaps']

    for i, attribute in enumerate(attributes):
        attr_value = getattr(old_video, attribute)
        if attr_value == True:
            logger.debug(attribute)
            existentFile[processes[i]] = '1'
        elif attr_value is False:
            existentFile[processes[i]] = '0'
        elif attr_value is None:
            existentFile[processes[i]] = '-1'
    return existentFile

def getExistentFiles(video, processes):
    """get processes already computed in a previous session
    preprocessing: segmentation, fragmentation and creation of blobs and individual/global fragments
    knowledge_transfer: knowledge transferred from a model trained on a different video
    first_accumulation: first accumulation attempt
    pretraining: building the filters in a global-identity-agnostic way
    second_accumulation: accumulation by transferring knowledge from pre-training
    assignment: assignment of the idenitity to each individual fragment
    solving_duplications: solve eventual identity duplications
    crossings: assign identity to single animals during occlusions
    trajectories: compute the individual trajectories
    """
    existentFile = {name:'-1' for name in processes}
    old_video = None
    if os.path.isdir(video._previous_session_folder):
        logger.debug("loading old video object from get existent files")
        if os.path.isfile(os.path.join(video._previous_session_folder, 'video_object.npy')):
            old_video = np.load(os.path.join(video._previous_session_folder, 'video_object.npy')).item()
            video.use_previous_knowledge_transfer_decision = old_video.use_previous_knowledge_transfer_decision
            logger.info("old video loaded")
        else:
            logger.info("The folder %s is empty. The tracking cannot be restored." %video._previous_session_folder)
            video.use_previous_knowledge_transfer_decision = False
            return existentFile, old_video
        old_video = check_and_change_video_path(video,old_video)
        existentFile = set_load_previous_dict(old_video, processes, existentFile)
    else:
        video.use_previous_knowledge_transfer_decision = False

    return existentFile, old_video


def get_existent_preprocessing_steps(old_video, listNames):
    """
    get processes already computed in a previous session
    """
    existentFile = {name:'0' for name in listNames}
    if old_video.bkg is not None:
        print('has bkg')
        existentFile['bkg'] = '1'
    if old_video.ROI is not None:
        print('has roi')
        existentFile['ROI'] = '1'
    if hasattr(old_video, 'resolution_reduction'):
        if old_video.resolution_reduction is not None:
            print('has resolution_reduction')
            existentFile['resolution_reduction'] = '1'
    return existentFile

def getExistentFiles_library(path, listNames, segmPaths):
    """
    get processes already computed in a previous session
    """
    existentFile = {name:'0' for name in listNames}
    video = os.path.basename(path)
    folder = os.path.dirname(path)

    # createFolder(path)

    #count how many videos we have
    numSegments = len(segmPaths)

    filename, extension = os.path.splitext(video)
    subFolders = glob.glob(folder +"/*/")

    srcSubFolder = folder + '/preprocessing/'
    for name in listNames:
        if name == 'segmentation':
            segDirname = srcSubFolder + name
            if os.path.isdir(segDirname):
                print 'Segmentation folder exists'
                numSegmentedVideos = len(glob.glob1(segDirname,"*.pkl"))
                print
                if numSegmentedVideos == numSegments:
                    print 'The number of segments and videos is the same'
                    existentFile[name] = '1'
        else:
            extensions = ['.pkl', '.hdf5']
            for ext in extensions:
                fullFileName = srcSubFolder + '/' + name + ext
                if os.path.isfile(fullFileName):
                    existentFile[name] = '1'

    return existentFile, srcSubFolder


def createFolder(path, name = '', timestamp = False):

    folder = os.path.dirname(path)
    folderName = folder +'/preprocessing'
    if timestamp :
        ts = '{:%Y%m%d%H%M%S}_'.format(datetime.datetime.now())
        folderName = folderName + '_' + ts

    if os.path.isdir(folderName):
        print 'Preprocessing folder exists'
        subFolder = folderName + '/segmentation'
        if os.path.isdir(subFolder):
            print 'Segmentation folder exists'
        else:
            os.makedirs(subFolder)
            print subFolder + ' has been created'
    else:
        os.makedirs(folderName)
        print folderName + ' has been created'
        subFolder = folderName + '/segmentation'
        os.makedirs(subFolder)
        print subFolder + ' has been created'

def createSessionFolder(videoPath):
    def getLastSession(subFolders):
        if len(subFolders) == 0:
            lastIndex = 0
        else:
            subFolders = natural_sort(subFolders)[::-1]
            lastIndex = int(subFolders[0].split('_')[-1])
        return lastIndex

    video = os.path.basename(videoPath)
    folder = os.path.dirname(videoPath)
    filename, extension = os.path.splitext(video)
    subFolder = folder + '/CNN_models'
    subSubFolders = glob.glob(subFolder +"/*")
    lastIndex = getLastSession(subSubFolders)
    sessionPath = subFolder + '/Session_' + str(lastIndex + 1)
    os.makedirs(sessionPath)
    print 'You just created ', sessionPath
    figurePath = sessionPath + '/figures'
    os.makedirs(figurePath)
    print 'You just created ', figurePath

    return sessionPath, figurePath
