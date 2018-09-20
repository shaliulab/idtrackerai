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
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (F.R.-F. and M.G.B. contributed equally to this work. Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)


from __future__ import division, absolute_import, print_function
from itertools import groupby
import os
import glob
import re
import numpy as np
import multiprocessing
import sys
import matplotlib
from matplotlib import cm
import subprocess

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.py_utils")

### Git utils ###
def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except:
        return 'not in a git repository'

### MKL
def set_mkl_to_single_thread():
    logger.info('Setting MKL library to use single thread')
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
def set_mkl_to_multi_thread():
    logger.info('Setting MKL library to use multiple threads')
    os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
    os.environ['MKL_DYNAMIC'] = 'TRUE'

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
    video_path = os.path.basename(path)
    filename, extension = os.path.splitext(video_path)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == '_1':
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

def get_spaced_colors_util(n, norm = False, black = True, cmap = 'jet'):
    RGB_tuples = cm.get_cmap(cmap)
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

def check_and_change_video_path(video_object, old_video):
    current_video_folder = os.path.split(video_object.video_path)[0]
    old_video_folder = os.path.split(old_video.video_path)[0]
    old_video_session_name = old_video.session_folder
    print("in check and change video path")
    print("current_video_folder: ", current_video_folder)
    print("old_video_folder: ", old_video_folder)
    print("old_video_session_name: ", old_video_session_name)
    if current_video_folder != old_video_folder:
        print("updating attributes")
        attributes_to_modify = {key: getattr(old_video, key) for key in old_video.__dict__
        if isinstance(getattr(old_video, key), basestring)
        and old_video_folder in getattr(old_video, key) }

        for key in attributes_to_modify:
            new_value = attributes_to_modify[key].replace(old_video_folder, current_video_folder)
            setattr(old_video, key, new_value)

        if old_video.paths_to_video_segments is not None and len(old_video.paths_to_video_segments) != 0:
            logger.info("Updating paths_to_video_segments")
            new_paths_to_video_segments = []
            for path in old_video.paths_to_video_segments:
                new_path = os.path.join(old_video.video_folder, os.path.split(path)[1])
                new_paths_to_video_segments.append(new_path)
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
    print("in set load previous dict")
    attributes = [ 'has_been_preprocessed',
                    'first_accumulation_finished',
                    'has_been_pretrained', 'second_accumulation_finished',
                    'has_been_assigned',
                    'has_crossings_solved', 'has_trajectories',
                    'has_trajectories_wo_gaps']

    # gui_processes = ['preprocessing','protocols1_and_2', 'protocol3_pretraining',
    #                 'protocol3_accumulation', 'residual_identification',
    #                 'post_processing']
    processes_to_attributes = { 'preprocessing' : ['has_been_preprocessed'],
            'protocols1_and_2' : ['first_accumulation_finished'],
            'protocol3_pretraining' : ['has_been_pretrained'],
            'protocol3_accumulation' : ['second_accumulation_finished'],
            'residual_identification' : ['has_been_assigned'],
            'post_processing' : ['has_crossings_solved', 'has_trajectories',
                                'has_trajectories_wo_gaps']}
    for process in processes:
        attributes = processes_to_attributes[process]
        attributes_values = []
        print(process)
        for attribute in attributes:
            print(attribute, getattr(old_video, attribute))
            attributes_values.append(getattr(old_video, attribute))
        if None in attributes_values:
            existentFile[process] = '-1'
        elif all(attributes_values):
            logger.debug(attribute)
            existentFile[process] = '1'
        else:
            existentFile[process] = '0'

    return existentFile

def getExistentFiles(video_object, processes):
    """get processes already computed in a previous session
    preprocessing: segmentation, fragmentation and creation of blobs and individual/global fragments
    knowledge_transfer: knowledge transferred from a model trained on a different video_object
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
    if os.path.isdir(video_object._previous_session_folder):
        logger.debug("loading old video object from get existent files")
        if os.path.isfile(os.path.join(video_object._previous_session_folder, 'video_object.npy')):
            print(video_object._previous_session_folder)
            old_video = np.load(os.path.join(video_object._previous_session_folder, 'video_object.npy')).item()
            old_video.update_paths(os.path.join(video_object._previous_session_folder, 'video_object.npy'))
            logger.info("old video_object loaded")
        else:
            logger.info("The folder %s is empty. The tracking cannot be restored." %video_object._previous_session_folder)
            return existentFile, old_video

        old_video = check_and_change_video_path(video_object,old_video)
        existentFile = set_load_previous_dict(old_video, processes, existentFile)

    return existentFile, old_video


def get_existent_preprocessing_steps(old_video, listNames):
    """
    get processes already computed in a previous session
    """
    existentFile = {name:'0' for name in listNames}
    if old_video.bkg is not None:
        existentFile['bkg'] = '1'
    if old_video.ROI is not None:
        existentFile['ROI'] = '1'
    if hasattr(old_video, 'resolution_reduction'):
        if old_video.resolution_reduction is not None:
            existentFile['resolution_reduction'] = '1'
    return existentFile
