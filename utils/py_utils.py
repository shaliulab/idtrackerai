from itertools import groupby
import os
import glob
import re
import datetime
import pandas as pd
import numpy as np
import Tkinter, tkSimpleDialog

### Dict utils ###
def getVarFromDict(dictVar,variableNames):
    ''' get variables from a standard python dictionary '''
    return [dictVar[v] for v in variableNames]

### Array utils ####
def maskArray(im1,im2,w1,w2):
    return np.add(np.multiply(im1,w1),np.multiply(im2,w2))

def uint8caster(im):
    return np.multiply(np.true_divide(im,np.max(im)),255).astype('uint8')

def flatten(l):
    ''' flatten a list of lists '''
    return [inner for outer in l for inner in outer]

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

def scanFolder(path):
    paths = [path]
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    # maybe write check on video extension supported by opencv2
    if filename[-2:] == '_1':
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

def saveFile(path, variabletoSave, name, time = 0):
    """
    All the input are strings!!!
    path: path to the first segment of the video
    name: string to add to the name of the video (wihtout timestamps)
    folder: path to the folder in which the file has to be stored
    """
    if os.path.exists(path)==False:
        raise ValueError("the video %s does not exist!" %path)
    video = os.path.basename(path)
    filename, extension = os.path.splitext(video)
    folder = os.path.dirname(path)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    # print 'subFolders', subFolders
    subFolder = subFolders[time]
    filename, extension = os.path.splitext(video)
    # we assume there's an underscore before the timestamp
    if name == 'segment' or name == 'segmentation':
        nSegment = filename.split('_')[-1]# and before the number of the segment
        filename = filename.split('_')[0] + '_' + nSegment + '.pkl'
        pd.to_pickle(variabletoSave, subFolder +'/segmentation/'+ filename)
    else:
        filename = filename.split('_')[0] + '_' + name + '.pkl'
        pd.to_pickle(variabletoSave, subFolder + '/'+ filename)
    print 'you just saved: ',subFolder + filename



def loadFile(path, name, time=0):
    """
    loads a pickle. path is the path of the video, while name is a string in the
    set {}
    """
    video = os.path.basename(path)
    folder = os.path.dirname(path)
    filename, extension = os.path.splitext(video)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    # print 'subFolders from loadFile ',subFolders
    subFolder = subFolders[time]

    if name  == 'segmentation':
        # print 'i am here'
        nSegment = filename.split('_')[-1]
        filename = filename.split('_')[0] + '_' + nSegment + '.pkl'
        # print filename
        # print subFolder
        # print nSegment
        return pd.read_pickle(subFolder + 'segmentation/' + filename ), nSegment
    else:
        filename = filename.split('_')[0] + '_' + name + '.pkl'
        return pd.read_pickle(subFolder + filename )


def createFolder(path, name = '', timestamp = False):

    ts = '{:%Y%m%d%H%M%S}_'.format(datetime.datetime.now())
    name = ts + name

    folder = os.path.dirname(path)
    folderName = folder +'/'+ name + '/segmentation'
    os.makedirs(folderName) # create a folder

    # folderName = folderName
    # os.makedirs(folderName) # create a folder

    print folderName + ' has been created'

"""
Display messages and errors
"""
def getInput(name,text):
    root = Tkinter.Tk() # dialog needs a root window, or will create an "ugly" one for you
    root.withdraw() # hide the root window
    password = tkSimpleDialog.askstring(name, text, parent=root)
    root.destroy() # clean up after yourself!
    return password

def displayMessage(title,message):
    window = Tk()
    window.wm_withdraw()

    #centre screen message
    window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))
    tkMessageBox.showinfo(title=title, message=message)

def displayError(title, message):
    #message at x:200,y:200
    window = Tk()
    window.wm_withdraw()

    window.geometry("1x1+200+200")#remember its .geometry("WidthxHeight(+or-)X(+or-)Y")
    tkMessageBox.showerror(title=title,message=message,parent=window)


# a = 1
# createFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', 'test')
# saveFile('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', a, 'test', 'test', addSegNum = False)
# b = loadFile('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', 'test')
