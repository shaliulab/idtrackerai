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
    fig1, ax = plt.subplots(1,1)
    fig1.set_size_inches((screen_x/100/2,screen_y/100*2/3))
    plt.subplots_adjust(left=None, bottom=.2, right=None, top=None,
                wspace=None, hspace=None)
    scales = []
    shapes = []
    for i, path_to_video_to_plot in enumerate(paths_to_videos_to_plot):
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

            ######## Gamma fit to number of images in individual fragments ########
            shape, loc, scale = gamma.fit(individual_fragments_lenghts, floc = 0.99)
            shapes.append(shape)
            scales.append(scale)
            print("shape %.2f, loc %.2f, scale %.2f" %(shape, loc, scale))
            gamma_fitted = gamma(shape, loc, scale)
            gamma_values = gamma_fitted.rvs(len(individual_fragments_lenghts))
            gamma_fitted_logpdf = pdf2logpdf(gamma_fitted.pdf)
            ######### number of images in individual fragments ########
            nbins = 25
            logbins = np.linspace(np.log10(min), np.log10(max), nbins)
            logbins2 = np.linspace(np.log10(min), np.log10(max), 100)
            fit_h = ax.plot(logbins2, gamma_fitted_logpdf(np.power(10,logbins2)),
                    label = 'Fit to ' + r'$\Gamma (k, \theta)$',
                    color = 'k',
                    linestyle = '--',
                    linewidth = 3)
            n, bins = np.histogram(np.log10(individual_fragments_lenghts), bins = logbins, normed = True)
            video_h = ax.plot(bins[:-1] + np.diff(bins)[0]/2, n, label = 'zebrafish videos', color = 'g', linewidth = 3, linestyle = '-')
            if i == 0:
                plt.legend(handles = [video_h[0], fit_h[0]], fontsize = 22)
        else:
            print('video_object_path does not exist')
    ax.set_xlim((0,4))
    ax.set_ylim((0, 1.2))
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels([1, 10,100,1000,10000])
    ax.set_xlabel('Number of images per individual fragment', fontsize = 25)
    ax.set_ylabel('PDF', fontsize = 25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    k_text = r'k = %.2f $\pm$ %.2f' %(np.mean(shapes), np.std(shapes))
    theta_text = r'$\theta$ = %.2f $\pm$ %.2f' %(np.mean(scales), np.std(scales))
    ax.text(2.7,.9,k_text, fontsize = 22)
    ax.text(2.7,.8,theta_text, fontsize = 22)
    sns.despine(ax = ax, left = False, bottom = False, right = True, top = True)
    fig1.savefig(os.path.join(tracked_videos_folder, 'distributions_individual_fragments_zebrafish.pdf'), transparent = True)
