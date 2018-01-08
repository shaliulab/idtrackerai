from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('../')
sys.path.append('./utils')
sys.path.append('./library')

import numpy as np
import collections
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
import seaborn as sns
import pandas as pd
from pprint import pprint
from scipy.stats import gamma
from scipy import stats

from py_utils import get_spaced_colors_util
from library_utils import LibraryJobConfig
from video import Video

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

if __name__ == '__main__':
    folder_to_videos = '/home/chronos/Desktop/IdTrackerDeep/videos/videos_info_for_fig_3'
    group_sizes = [10, 60, 100]
    conditions = ['first', 'second']
    repetitions = ['group1', 'group2', 'group3']
    nbins = 25
    min = 1
    max = 10000

    plt.ion()
    sns.set_style("ticks")
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1, ax_arr1 = plt.subplots(3, 3, sharey = True)
    fig2, ax_arr2 = plt.subplots(3, 3, sharey = True)
    fig1.set_size_inches((screen_x/100,screen_y/100))
    fig2.set_size_inches((screen_x/100,screen_y/100))

    for i, group_size in enumerate(group_sizes):

        for condition in conditions:

            for j, repetition in enumerate(repetitions):
                ax = ax_arr1[j, i] if condition == 'first' else ax_arr2[j, i]
                fig = fig1 if condition == 'first' else fig2
                video_object_path = os.path.join(folder_to_videos, str(group_size) + 'fish', condition, repetition, 'video_object.npy')
                light_list_of_fragments_path = os.path.join(folder_to_videos, str(group_size) + 'fish', condition, repetition, 'light_list_of_fragments.npy')
                print(video_object_path)
                if os.path.isfile(video_object_path):
                    print("exists")
                    video = np.load(video_object_path).item()
                    try:
                        individual_fragments_lenghts = video.individual_fragments_lenghts
                    except:
                        print("does not have it")
                        if os.path.isfile(light_list_of_fragments_path):
                            llfs = np.load(light_list_of_fragments_path)
                            individual_fragments_lenghts = [len(f['areas']) for f in llfs if f['is_an_individual']]
                            video.individual_fragments_lenghts = individual_fragments_lenghts
                            print('saving video')
                            np.save(video_object_path, video)

                    ######## Gamma fit to number of images in individual fragments ########
                    shape, loc, scale = gamma.fit(individual_fragments_lenghts, floc = 0.99)
                    print("shape %.2f, loc %.2f, scale %.2f" %(shape, loc, scale))
                    gamma_fitted = gamma(shape, loc, scale)
                    gamma_values = gamma_fitted.rvs(len(individual_fragments_lenghts))
                    gamma_fitted_logpdf = pdf2logpdf(gamma_fitted.pdf)
                    ######### number of images in individual fragments ########
                    nbins = 25
                    logbins = np.linspace(np.log10(min), np.log10(max), nbins)
                    n, _, _  = ax.hist(np.log10(individual_fragments_lenghts), bins = logbins, normed = True, label = 'exp. data')
                    logbins2 = np.linspace(np.log10(min), np.log10(max), 100)
                    ax.plot(logbins2, gamma_fitted_logpdf(np.power(10,logbins2)), label = 'fit')
                    ax.set_xlim((np.log10(min), np.log10(max)))
                    text = 'shape = %.2f, scale = %.2f' %(shape, scale)
                    ax.text(.5,.6,text)
                    ax.set_ylim((0, np.max(n)))
                    ax.set_xlim((np.log10(min),np.log10(max)))

                if j == 0:
                    ax.set_title(str(group_size) + ' individuals', fontsize = 15)
                if j == 2:
                    ax.set_xticks([0,1,2,3,4])
                    ax.set_xticklabels([1, 10,100,1000,10000], fontsize = 12)
                    ax.set_xlabel('images per individal fragment', fontsize = 12)
                else:
                    ax.set_xticks([0, 1, 2, 3, 4])
                    ax.set_xticklabels([])
                if i == 0:
                    ax.set_ylabel('PDF', fontsize = 12)
                if j == 2 and i == 2:
                    ax.legend(frameon = True)
                sns.despine(ax = ax, left = False, bottom = False, right = True, top = True)

    fig1.savefig(os.path.join(folder_to_videos, 'distributions_individual_fragments_first.pdf'), transparent = True)
    fig2.savefig(os.path.join(folder_to_videos, 'distributions_individual_fragments_second.pdf'), transparent = True)
