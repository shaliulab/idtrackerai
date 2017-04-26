import pandas as pd
from matplotlib import pyplot as plt

def showMiniframes(path,frame):

    df = pd.read_pickle(path)
    plt.ion()
    for i, miniframe in enumerate(df.loc[frame,'miniFrames']):
        plt.figure(i)
        plt.imshow(miniframe,interpolation='none',cmap='gray')

    plt.show()
