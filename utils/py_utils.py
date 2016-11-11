from itertools import groupby
import os
import glob
import re
import datetime
import pandas as pd
import numpy as np
import Tkinter, tkSimpleDialog, tkFileDialog,tkMessageBox
from Tkinter import *
import shutil

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

# def saveFile(path, variabletoSave, name, time = 0):
#     """
#     All the input are strings!!!
#     path: path to the first segment of the video
#     name: string to add to the name of the video (wihtout timestamps)
#     folder: path to the folder in which the file has to be stored
#     """
#     if os.path.exists(path)==False:
#         raise ValueError("the video %s does not exist!" %path)
#     video = os.path.basename(path)
#     filename, extension = os.path.splitext(video)
#     folder = os.path.dirname(path)
#     subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
#     subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
#     # print 'subFolders Saver', subFolders
#     subFolder = subFolders[time]
#     filename, extension = os.path.splitext(video)
#     # we assume there's an underscore before the timestamp
#     if name == 'segment' or name == 'segmentation':
#         nSegment = filename.split('_')[-1]# and before the number of the segment
#         filename = filename.split('_')[0] + '_' + nSegment + '.hdf5'
#         pd.to_hdf(variabletoSave, subFolder +'/segmentation/'+ filename)
#     else:
#         filename = filename.split('_')[0] + '_' + name + '.hdf5'
#         pd.to_hdf(variabletoSave, subFolder + '/'+ filename)
#     print 'you just saved: ',subFolder + filename



# def loadFile(path, name, time=0):
#     """
#     loads a pickle. path is the path of the video, while name is a string in the
#     set {}
#     """
#     video = os.path.basename(path)
#     folder = os.path.dirname(path)
#     filename, extension = os.path.splitext(video)
#     subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
#     subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
#     # print 'subFolders Loader', subFolders
#     # print 'subFolders from loadFile ',subFolders
#     subFolder = subFolders[time]
#
#     if name  == 'segmentation':
#         # print 'i am here'
#         nSegment = filename.split('_')[-1]
#         filename = filename.split('_')[0] + '_' + nSegment + '.hdf5'
#         # print filename
#         # print subFolder
#         # print nSegment
#         return pd.read_hdf(subFolder + 'segmentation/' + filename ), nSegment
#     else:
#         filename = filename.split('_')[0] + '_' + name + '.hdf5'
#         return pd.read_hdf(subFolder + filename )

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
    subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
    # print 'subFolders Saver', subFolders
    subFolder = subFolders[time]
    filename, extension = os.path.splitext(video)
    # we assume there's an underscore before the timestamp
    if name == 'segment' or name == 'segmentation':
        nSegment = filename.split('_')[-1]# and before the number of the segment
        filename = 'segm_' + nSegment + '.hdf5'
        variabletoSave.to_hdf(subFolder +'/segmentation/'+ filename,name)
    else:
        filename = name + '.hdf5'
        if isinstance(variabletoSave, dict):
            variabletoSave = pd.DataFrame.from_dict(variabletoSave,orient='index')
        elif not isinstance(variabletoSave, pd.DataFrame):
            variabletoSave = pd.DataFrame(variabletoSave)
        variabletoSave.to_hdf(subFolder + filename,name)

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
    subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
    # print 'subFolders Loader', subFolders
    # print 'subFolders from loadFile ',subFolders
    subFolder = subFolders[time]
    print 'Loading ' + name + ' from subfolder ', subFolder

    if name  == 'segmentation':
        # print 'i am here'
        nSegment = filename.split('_')[-1]
        filename = 'segm_' + nSegment + '.hdf5'
        return pd.read_hdf(subFolder + 'segmentation/' + filename ), nSegment
    else:
        filename = name + '.hdf5'
        return pd.read_hdf(subFolder + filename )

def copyExistentFiles(path, listNames, time=1):
    """
    Load data from previous session (time = 1 means we are looking back of one step)
    """
    existentFile = {name:'0' for name in listNames}
    # existentFile = dict()
    createFolder(path, name = '', timestamp = False)
    video = os.path.basename(path)
    folder = os.path.dirname(path)
    #count how many videos we have
    numVideos = len(glob.glob1(folder,"*.avi"))

    filename, extension = os.path.splitext(video)
    subFolders = natural_sort(glob.glob(folder +"/*/"))[::-1]
    subFolders = [subFolder for subFolder in subFolders if subFolder.split('/')[-2][0].isdigit()]
    print subFolders
    if len(subFolders) <= 1:
        srcSubFolder = 'There is not previous subFolder'
        pass
    else:
        srcSubFolder = subFolders[time]
        dstSubFolder = subFolders[time-1]
        for name in listNames:
            if name == 'segmentation':
                segDirname = srcSubFolder + '/' + name
                if os.path.isdir(segDirname):
                    print 'Segmentation folder exists'
                    numSegmentedVideos = len(glob.glob1(segDirname,"*.hdf5"))
                    print
                    if numSegmentedVideos == numVideos:
                        print 'The number of segments and videos is the same'
                        existentFile[name] = '1'
                        dstSubFolderSeg = dstSubFolder + '/segmentation'
                        srcFiles = os.listdir(segDirname)
                        for fileName in srcFiles:
                            fullFileName = os.path.join(segDirname, fileName)
                            if (os.path.isfile(fullFileName)):
                                shutil.copy(fullFileName, dstSubFolderSeg)
                        # if segmentation is copyed we also copy frameIndices and videoInfo
                        fullFileName = srcSubFolder + '/frameIndices.hdf5'
                        if os.path.isfile(fullFileName):
                            shutil.copy(fullFileName, dstSubFolder)
                        fullFileName = srcSubFolder + '/videoInfo.hdf5'
                        if os.path.isfile(fullFileName):
                            shutil.copy(fullFileName, dstSubFolder)

            else:
                if name is 'fragmentation':
                    segDirname = srcSubFolder + 'segmentation'
                    if os.path.isdir(segDirname):
                        srcFiles = os.listdir(segDirname)
                        if os.path.isdir(segDirname) and len(srcFiles)!=0:
                            df,_ = loadFile(path, 'segmentation', time=0)
                            if 'permutation' in list(df.columns):
                                existentFile[name] = '1'

                else:
                    fullFileName = srcSubFolder + '/' + name + '.hdf5'
                    if os.path.isfile(fullFileName):
                        existentFile[name] = '1'
                        shutil.copy(fullFileName, dstSubFolder)
                    if name is 'ROI':
                        fullFileName = srcSubFolder + '/centers.hdf5'
                        if os.path.isfile(fullFileName):
                            shutil.copy(fullFileName, dstSubFolder)


    return existentFile, srcSubFolder

def createFolder(path, name = '', timestamp = False):

    ts = '{:%Y%m%d%H%M%S}_'.format(datetime.datetime.now())
    name = ts + name

    folder = os.path.dirname(path)
    folderName = folder +'/'+ name + '/segmentation'
    os.makedirs(folderName) # create a folder

    # folderName = folderName
    # os.makedirs(folderName) # create a folder

    print folderName + ' has been created'
#
# createFolder('/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf/group_1_camera_1/group_1_camera_1_20160508T094501_1.avi', name = '', timestamp = False)
# copyExistentFiles('/home/lab/Desktop/TF_models/IdTracker/data/library/25dpf/group_1_camera_1/group_1_camera_1_20160508T094501_1.avi', ['mask', 'centers', 'bkg','segmentation'], time=1)

"""
Display messages and errors
"""
# def selectOptions(optionsList):
#     opt = []
#     def chkbox_checked():
#         for ix, item in enumerate(cb):
#             if cb_v[ix].get() is '0':
#                 opt[ix]=('0')
#             else:
#                 opt[ix]=('1')
#         print opt
#     root = Tk()
#     cb = []
#     cb_v = []
#     for ix, text in enumerate(optionsList):
#         cb_v.append(StringVar())
#         off_value=0  #whatever you want it to be when the checkbutton is off
#         cb.append(Checkbutton(root, text=text, onvalue=text,offvalue=off_value,
#                                  variable=cb_v[ix],
#                                  command=chkbox_checked))
#         cb[ix].grid(row=ix, column=0, sticky='w')
#         opt.append(off_value)
#         cb[-1].deselect() #uncheck the boxes initially.
#     label = Label(root, width=20)
#     label.grid(row=ix+1, column=0, sticky='w')
#     b1 = Button(root,text = 'Quit', command= root.quit)
#     root.mainloop()
#     return opt

def selectOptions(optionsList, optionsDict=None, text="Select preprocessing options:  "):
    master = Tk()
    if optionsDict==None:
        optionsDict = {el:'1' for el in optionsList}
    def createCheckBox(name,i):
        var = IntVar()
        Checkbutton(master, text=name, variable=var).grid(row=i+1, sticky=W)
        return var

    Label(master, text=text).grid(row=0, sticky=W)
    variables = []
    for i, opt in enumerate(optionsList):
        if optionsDict[opt] == '1':
            var = createCheckBox(opt,i)
            variables.append(var)
            var.set(optionsDict[opt])
        else:
            Label(master, text= '     ' + opt).grid(row=i+1, sticky=W)
            var = IntVar()
            var.set(0)
            variables.append(var)

    Button(master, text='Ok', command=master.quit).grid(row=i+2, sticky=W, pady=4)
    mainloop()
    varValues = []
    for var in variables:
        varValues.append(var.get())
    optionsDict = dict((key, value) for (key, value) in zip(optionsList, varValues))
    master.destroy()
    return optionsDict

def selectFile():
    root = Tkinter.Tk()
    root.withdraw()
    filename = tkFileDialog.askopenfilename()
    root.destroy()
    return filename

def selectDir(initialDir):
    root = Tkinter.Tk()
    root.withdraw()
    dirName = tkFileDialog.askdirectory(initialdir = initialDir)
    root.destroy()
    return dirName

def getInput(name,text):
    root = Tkinter.Tk() # dialog needs a root window, or will create an "ugly" one for you
    root.withdraw() # hide the root window
    inputString = tkSimpleDialog.askstring(name, text, parent=root)
    root.destroy() # clean up after yourself!
    return inputString.lower()

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
