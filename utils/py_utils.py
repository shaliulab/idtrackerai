from __future__ import division
# Import standard libraries
import os
import glob
import re
import datetime
import numpy as np

# Import third party libraries
from itertools import groupby
import pandas as pd
import cPickle as pickle


### Dict utils ###
def getVarFromDict(dictVar,variableNames):
    ''' get variables from a standard python dictionary '''
    return [dictVar[v] for v in variableNames]


def maskArray(im1,im2,w1,w2):
    return np.add(np.multiply(im1,w1),np.multiply(im2,w2))

def uint8caster(im):
    return np.multiply(np.true_divide(im,np.max(im)),255).astype('uint8')

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
    else:
        paths = natural_sort(glob.glob(folder + "/" + filename[:-1] + "*" + extension))
    return paths

def get_spaced_colors_util(n,norm=False):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(100, max_value, interval)]
    if norm:
        rgbcolorslist = [(int(i[4:], 16)/256., int(i[2:4], 16)/256., int(i[:2], 16)/256.) for i in colors]
    else:
        rgbcolorslist = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
    black = (0., 0., 0.)
    rgbcolorslist.insert(0, black)
    return rgbcolorslist

def saveFile(path, variabletoSave, name, hdfpkl = 'hdf',sessionPath = ''):
    import cPickle as pickle
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
    filename, extension = os.path.splitext(video)

    if name == 'segment' or name == 'segmentation':
        subfolder = '/preprocessing/segmentation/'
        nSegment = filename.split('_')[-1]# and before the number of the segment
        if hdfpkl == 'hdf':
            filename = 'segm_' + nSegment + '.hdf5'
            pathToSave = folder + subfolder + filename
            variabletoSave.to_hdf(pathToSave,name)
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
            filename = name + '.hdf5'
            if isinstance(variabletoSave, dict):
                variabletoSave = pd.DataFrame.from_dict(variabletoSave,orient='index')
            elif not isinstance(variabletoSave, pd.DataFrame):
                variabletoSave = pd.DataFrame(variabletoSave)
            pathToSave = folder + subfolder + filename
            variabletoSave.to_hdf(pathToSave,name)
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

    if name  == 'segmentation':
        subfolder = '/preprocessing/segmentation/'
        nSegment = filename.split('_')[-1]
        if hdfpkl == 'hdf':
            filename = 'segm_' + nSegment + '.hdf5'
            return pd.read_hdf(folder + subfolder + filename ), nSegment
        elif hdfpkl == 'pkl':
            filename = 'segm_' + nSegment + '.pkl'
            return pickle.load(open(folder + subfolder + filename) ,'rb'), nSegmen
    elif name == 'statistics':
        filename = 'statistics.pkl'
        return pickle.load(open(sessionPath + '/' + filename,'rb') )
    else:
        subfolder = '/preprocessing/'
        if hdfpkl == 'hdf':
            filename = name + '.hdf5'
            return pd.read_hdf(folder + subfolder + filename )
        elif hdfpkl == 'pkl':
            filename = name + '.pkl'
            return pickle.load(open(folder + subfolder + filename,'rb') )

    print 'You just loaded ', folder + subfolder + filename

def getExistentFiles(path, listNames):
    """
    get processes already computed in a previous session
    """
    existentFile = {name:'0' for name in listNames}
    video = os.path.basename(path)
    folder = os.path.dirname(path)

    createFolder(path)

    #count how many videos we have
    numVideos = len(glob.glob1(folder,"*.avi"))

    filename, extension = os.path.splitext(video)
    subFolders = glob.glob(folder +"/*/")

    srcSubFolder = folder + '/preprocessing/'
    for name in listNames:
        if name == 'segmentation':
            segDirname = srcSubFolder + name
            if os.path.isdir(segDirname):
                print 'Segmentation folder exists'
                numSegmentedVideos = len(glob.glob1(segDirname,"*.hdf5"))
                print
                if numSegmentedVideos == numVideos:
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

def getMultipleInputs(winTitle, inputTexts):
    #Gui Things
    def retrieve_inputs():
        global inputs
        inputs = [var.get() for var in variables]
        window.destroy()
        return inputs
    window = Tk()
    window.title(winTitle)
    variables = []


    for inputText in inputTexts:
        text = Label(window, text =inputText)
        guess = Entry(window)
        variables.append(guess)
        text.pack()
        guess.pack()
    finished = Button(text="ok", command=retrieve_inputs)
    finished.pack()
    window.mainloop()

    return inputs

# inputs = getMultipleInputs('ciccio',['p','ccio', 'pagliaccio'])
# print inputs

# a = 1
# createFolder('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', 'test')
# saveFile('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', a, 'test', 'test', addSegNum = False)
# b = loadFile('../Cafeina5peces/Caffeine5fish_20140206T122428_1.avi', 'test')
