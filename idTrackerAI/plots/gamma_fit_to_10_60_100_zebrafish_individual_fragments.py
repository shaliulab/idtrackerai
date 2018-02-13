from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('./')
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
from video import Video

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

paths_to_videos_to_plot = ['10_fish_group4/first/session_20180122',
                    '10_fish_group5/first/session_20180131',
                    '10_fish_group6/first/session_20180202',
                    'idTrackerDeep_LargeGroups_1/100/First/session_20180102',
                    'idTrackerDeep_LargeGroups_1/60/First/session_20180108',
                    'idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_100/First/session_20180104',
                    'idTrackerDeep_LargeGroups_2/TU20170307/numberIndivs_60/First/session_20171221',
                    'idTrackerDeep_LargeGroups_3/100fish/First/session_02122017',
                    'idTrackerDeep_LargeGroups_3/60fish/First/session_20171225']

if __name__ == '__main__':
    path_to_results_hard_drive = '/media/atlas/ground_truth_results_backup'
    tracked_videos_folder = os.path.join(path_to_results_hard_drive, 'tracked_videos')
    nbins = 25
    min = 1
    max = 10000

    plt.ion()
    sns.set_style("ticks")
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig1, ax_arr1 = plt.subplots(3, 3, sharey = True)
    fig1.set_size_inches((screen_x/100,screen_y/100))

    rows = {10: 0, 60: 0, 100: 0}
    for path_to_video_to_plot in paths_to_videos_to_plot:
        video_object_path = os.path.join(tracked_videos_folder, path_to_video_to_plot, 'video_object.npy')
        print(video_object_path)
        if os.path.isfile(video_object_path):
            video = np.load(video_object_path).item()
            video.update_paths(video_object_path)
            try:
                individual_fragments_lenghts = video.individual_fragments_lenghts
            except:
                print("does not have it")
                light_list_of_fragments_path = os.path.join(video.accumulation_folder, 'light_list_of_fragments.npy')
                if os.path.isfile(light_list_of_fragments_path):
                    llfs = np.load(light_list_of_fragments_path)
                    individual_fragments_lenghts = [len(f['areas']) for f in llfs if f['is_an_individual']]
                    video.individual_fragments_lenghts = individual_fragments_lenghts
                    print('saving video')
                    np.save(video_object_path, video)

            if video.number_of_animals == 10:
                column = 0
            elif video.number_of_animals == 60:
                column = 1
            elif video.number_of_animals == 100:
                column = 2
            row = rows[video.number_of_animals]
            ax = ax_arr1[row, column]
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
            rows[video.number_of_animals] +=1

            if row == 0:
                ax.set_title(str(video.number_of_animals) + ' individuals', fontsize = 15)
            if row == 2:
                ax.set_xticks([0,1,2,3,4])
                ax.set_xticklabels([1, 10,100,1000,10000], fontsize = 12)
                ax.set_xlabel('images per individal fragment', fontsize = 12)
            else:
                ax.set_xticks([0, 1, 2, 3, 4])
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel('PDF', fontsize = 12)
            if row == 2 and column == 2:
                ax.legend(frameon = True)
            sns.despine(ax = ax, left = False, bottom = False, right = True, top = True)

        else:
            print('video_object_path does not exist')
    fig1.savefig(os.path.join(tracked_videos_folder, 'distributions_individual_fragments_10_60_100_zebrafish.pdf'), transparent = True)
