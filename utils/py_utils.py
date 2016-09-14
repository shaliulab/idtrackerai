from itertools import groupby
import os
import glob
import re
### Dict utils ###
def getVarFromDict(dictVar,variableNames):
    ''' get variables from a standard python dictionary '''
    return [dictVar[v] for v in variableNames]

### Array utils ####
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
