import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    results_noPretrain_noAccum_noGaussian = pd.read_pickle('/home/rhea/Desktop/IdTrackerDeep/restructure/library/results_data_frame_correlated_onlytrtain1GF_portraits.pkl')
    results_noPretrain_noAccum_Gaussian = pd.read_pickle('/home/rhea/Desktop/IdTrackerDeep/restructure/library/results_data_frame_noPretrain_noAccum_100fish_3000frames_gaussian.pkl')
    results_noPretrain_Accum01_Gaussian = pd.read_pickle('/home/rhea/Desktop/IdTrackerDeep/restructure/library/results_data_frame_noPretrain_accum_100fish_3000frames_gaussian_certainty01.pkl' )
    results_noPretrain_Accum05_Gaussian = pd.read_pickle('/home/rhea/Desktop/IdTrackerDeep/restructure/library/results_data_frame_noPretrain_accum_100fish_3000frames_gaussian_certainty05.pkl' )
    results_Pretrain_noAccum_Gaussian = pd.read_pickle('/home/rhea/Desktop/IdTrackerDeep/restructure/library/results_data_frame_pretrain_noAccum_100fish_3000frames_gaussian.pkl' )

    ac0 = np.mean(np.reshape(np.asarray(results_noPretrain_noAccum_noGaussian[results_noPretrain_noAccum_noGaussian['group_size']==100].accuracy),(5,3)),axis=1)
    ac1 = np.mean(np.reshape(np.asarray(results_noPretrain_noAccum_Gaussian.accuracy),(5,3)),axis=1)
    ac2 = np.mean(np.reshape(np.asarray(results_noPretrain_Accum01_Gaussian.accuracy),(5,3)),axis=1)
    ac3 = np.mean(np.reshape(np.asarray(results_noPretrain_Accum05_Gaussian.accuracy),(5,3)),axis=1)
    ac4 = np.mean(np.reshape(np.asarray(results_Pretrain_noAccum_Gaussian.accuracy),(5,3)),axis=1)

    accs = zip(ac0, ac1, ac2, ac3, ac4)
    mean_frames_per_fragment = [50,100,250,500,1000]

    plt.ion()
    fig, ax = plt.subplots(1)

    for i, mean_frames in enumerate(mean_frames_per_fragment):
        ax.plot([0,1,2,3,4],accs[i],label = str(mean_frames_per_fragment[i]))

    ax.legend()
    ax.set_xlabel('condition')
    ax.set_ylabel('accuracy')
    plt.xticks([0,1,2,3,4], ['noPretain-noAccum-noGaussian','noPretain-noAccum-Gaussian', 'noPretrain-Accum-certainty0.1-Gaussian', 'noPretrain-Accum-certainty0.5-Gaussian', 'pretrain-noAccum-Gaussian'])
