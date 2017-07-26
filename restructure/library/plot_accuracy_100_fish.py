import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    results = pd.read_pickle('library/results_data_frame_all_2.pkl')
    tests_names = [ 'correlated_images_DEF_aaa_CNN0_noPretrain_noAccum_100fish_3000frames_gaussian',
                    'correlated_images_DEF_aaa_CNN0_noPretrain_Accum05_100fish_3000frames_gaussian',
                    'correlated_images_DEF_aaa_CNN0_Pretrain_Accum05_100fish_3000frames_gaussian']
    frames_per_fragment_conditions = list(results.frames_per_fragment.unique())
    num_frames_conditions = len(frames_per_fragment_conditions)
    num_repetitions = len(list(results.repetition.unique()))

    acc = np.ones((4,5,10))*np.nan
    for i, test_name in enumerate(tests_names):
        results_test = results[results.test_name == test_name]
        acc[i,:,:] = np.reshape(np.asarray(results_test.accuracy),(num_frames_conditions,num_repetitions))

    plt.ion()
    fig, ax = plt.subplots(1)

    import colorsys
    HSV_tuples = [(x*1.0/num_frames_conditions, 0.5, 0.5) for x in range(num_frames_conditions)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    epsilon = [-0.01 , -0.005, 0., 0.005, 0.01]
    epsilon = [-0.1 , -0.05, 0., 0.05, 0.1]
    epsilon = [0. , 0., 0., 0., 0.]
    for i, mean_frames in enumerate(frames_per_fragment_conditions):
        accuracies = np.squeeze(acc[:,i,:])
        print(accuracies)
        acc_mean = np.nanmean(accuracies,axis = 1)
        ax.plot(np.asarray([0,1,2]),acc_mean,label = str(mean_frames), color = np.asarray(RGB_tuples[i]))
        for j in range(10):
            ax.scatter(np.asarray([0,1,2])+epsilon[i], accuracies[:,j], color = np.asarray(RGB_tuples[i]), alpha = .3)

    ax.legend()
    ax.set_xlabel('condition')
    ax.set_ylabel('accuracy')
    plt.xticks([0,1,2], [ 'noPretain-noAccum',
                                'noPretrain-Accum',
                                'Pretrain-Accum'])
    ax.set_ylim(0.75,1.01)
