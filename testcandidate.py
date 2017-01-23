
indivFragsForTrainList = [(1,2,3,4), (2,3,4,5)]
idP2Frags = [(10,20,3,4), (2,5,4,5),(2,5,1,5),(2,5,4,1)]

candidates = idP2Frags
for frag in indivFragsForTrainList:
    candidates = [idP2Frag for idP2Frag in candidates if idP2Frag[2:4] != frag[2:4] ]
print candidates
