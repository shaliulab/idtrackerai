from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma, lognorm, exponweib
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib
font = {'family' : 'normal',
       'size'   : 18}
matplotlib.rc('font', **font)

def gamma_params(mean, std):
    a = (mean/std)**2
    s = std**2/mean
    return a, s

def mean_std_gamma(a,s):
    ''' a = shape (k in wikipedia)
        s = scale (theta in wikipedia)
    '''
    mean = a * s
    var = a * s**2
    std = np.sqrt(var)
    return mean, std

if __name__ == '__main__':
    nbins = 30

    print("Loading data ...")
    ''' shoaling '''
    shoaling = np.load('shoaling_individual_fragments_distribution.npy')
    shoaling =  filter(lambda x: x >= 1, shoaling)
    MIN = np.min(shoaling)
    MAX = np.max(shoaling)
    min_shoaling, max_shoaling, mean_shoaling, std_shoaling = np.min(shoaling), np.max(shoaling), np.mean(shoaling), np.std(shoaling)
    print("Shoaling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_shoaling, max_shoaling, mean_shoaling, std_shoaling))
    # hist_shoaling, bin_edges_shoaling = np.histogram(shoaling, bins = 10 ** np.linspace(np.log10(min_shoaling), np.log10(max_shoaling), nbins))
    hist_shoaling, bin_edges_shoaling = np.histogram(shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    # fit gamma
    a, loc, scale = gamma.fit(shoaling,loc=1)
    gamma_fitted = gamma(a, loc, scale)
    gamma_shoaling = gamma_fitted.rvs(len(shoaling))
    min_gamma_shoaling, max_gamma_shoaling, mean_gamma_shoaling, std_gamma_shoaling = np.min(gamma_shoaling), np.max(gamma_shoaling), np.mean(gamma_shoaling), np.std(gamma_shoaling)
    print("Schooling (gamma): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_shoaling, max_gamma_shoaling, mean_gamma_shoaling, std_gamma_shoaling))
    # hist_gamma_shoaling, bin_edges_gamma_shoaling = np.histogram(gamma_shoaling, bins = 10 ** np.linspace(np.log10(min_gamma_shoaling), np.log10(max_gamma_shoaling), nbins))
    hist_gamma_shoaling, bin_edges_gamma_shoaling = np.histogram(gamma_shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    # fit lognormal
    a, loc, scale = lognorm.fit(shoaling,loc=1)
    lognorm_fitted = lognorm(a, loc, scale)
    lognorm_shoaling = lognorm_fitted.rvs(len(shoaling))
    min_lognorm_shoaling, max_lognorm_shoaling, mean_lognorm_shoaling, std_lognorm_shoaling = np.min(lognorm_shoaling), np.max(lognorm_shoaling), np.mean(lognorm_shoaling), np.std(lognorm_shoaling)
    print("Schooling (lognorm): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_lognorm_shoaling, max_lognorm_shoaling, mean_lognorm_shoaling, std_lognorm_shoaling))
    # hist_lognorm_shoaling, bin_edges_lognorm_shoaling = np.histogram(lognorm_shoaling, bins = 10 ** np.linspace(np.log10(min_lognorm_shoaling), np.log10(max_lognorm_shoaling), nbins))
    hist_lognorm_shoaling, bin_edges_lognorm_shoaling = np.histogram(lognorm_shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    # fit weibull
    loc, scale = exponweib.fit_loc_scale(shoaling, 1, 1)
    exponweib_fitted = exponweib(1, 1, loc, scale)
    exponweib_shoaling = exponweib_fitted.rvs(len(shoaling))
    min_exponweib_shoaling, max_exponweib_shoaling, mean_exponweib_shoaling, std_exponweib_shoaling = np.min(exponweib_shoaling), np.max(exponweib_shoaling), np.mean(exponweib_shoaling), np.std(exponweib_shoaling)
    print("Schooling (exponweib): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_exponweib_shoaling, max_exponweib_shoaling, mean_exponweib_shoaling, std_exponweib_shoaling))
    # hist_exponweib_shoaling, bin_edges_exponweib_shoaling = np.histogram(exponweib_shoaling, bins = 10 ** np.linspace(np.log10(min_exponweib_shoaling), np.log10(max_exponweib_shoaling), nbins))
    hist_exponweib_shoaling, bin_edges_exponweib_shoaling = np.histogram(exponweib_shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    ''' schooling '''
    schooling = np.load('schooling_individual_fragments_distribution.npy')
    schooling =  filter(lambda x: x >= 1, schooling)
    min_schooling, max_schooling, mean_schooling, std_schooling = np.min(schooling), np.max(schooling), np.mean(schooling), np.std(schooling)
    print("Schooling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_schooling, max_schooling, mean_schooling, std_schooling))
    # hist_schooling, bin_edges_schooling = np.histogram(schooling, bins = 10 ** np.linspace(np.log10(min_schooling), np.log10(max_schooling), nbins))
    hist_schooling, bin_edges_schooling = np.histogram(schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    #fit gamma
    a, loc, scale= gamma.fit(schooling,loc=1)
    gamma_fitted = gamma(a, loc, scale)
    gamma_schooling = gamma_fitted.rvs(len(schooling))
    min_gamma_schooling, max_gamma_schooling, mean_gamma_schooling, std_gamma_schooling = np.min(gamma_schooling), np.max(gamma_schooling), np.mean(gamma_schooling), np.std(gamma_schooling)
    print("Schooling (gamma): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_schooling, max_gamma_schooling, mean_gamma_schooling, std_gamma_schooling))
    # hist_gamma_schooling, bin_edges_gamma_schooling = np.histogram(gamma_schooling, bins = 10 ** np.linspace(np.log10(min_gamma_schooling), np.log10(max_gamma_schooling), nbins))
    hist_gamma_schooling, bin_edges_gamma_schooling = np.histogram(gamma_schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    # fit lognormal
    a, loc, scale = lognorm.fit(schooling,loc=1)
    lognorm_fitted = lognorm(a, loc, scale)
    lognorm_schooling = lognorm_fitted.rvs(len(schooling))
    min_lognorm_schooling, max_lognorm_schooling, mean_lognorm_schooling, std_lognorm_schooling = np.min(lognorm_schooling), np.max(lognorm_schooling), np.mean(lognorm_schooling), np.std(lognorm_schooling)
    print("Schooling (lognorm): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_lognorm_schooling, max_lognorm_schooling, mean_lognorm_schooling, std_lognorm_schooling))
    # hist_lognorm_schooling, bin_edges_lognorm_schooling = np.histogram(lognorm_schooling, bins = 10 ** np.linspace(np.log10(min_lognorm_schooling), np.log10(max_lognorm_schooling), nbins))
    hist_lognorm_schooling, bin_edges_lognorm_schooling = np.histogram(lognorm_schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    # fit weibull
    loc, scale = exponweib.fit_loc_scale(schooling, 1, 1)
    exponweib_fitted = exponweib(1 , 1, loc, scale)
    exponweib_schooling = exponweib_fitted.rvs(len(schooling))
    min_exponweib_schooling, max_exponweib_schooling, mean_exponweib_schooling, std_exponweib_schooling = np.min(exponweib_schooling), np.max(exponweib_schooling), np.mean(exponweib_schooling), np.std(exponweib_schooling)
    print("Schooling (exponweib): min %.2f, max %.2f, mean %.2f, std %.2f" %(min_exponweib_schooling, max_exponweib_schooling, mean_exponweib_schooling, std_exponweib_schooling))
    # hist_exponweib_schooling, bin_edges_exponweib_schooling = np.histogram(exponweib_schooling, bins = 10 ** np.linspace(np.log10(min_exponweib_schooling), np.log10(max_exponweib_schooling), nbins))
    hist_exponweib_schooling, bin_edges_exponweib_schooling = np.histogram(exponweib_schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    ''' superSchooling1 '''
    a, s = gamma_params(85, 168)
    print("superSchooling1: a %.2f, s %.2f" %(a,s))
    gamma_fitted = gamma(.35, 1, 250)
    # gamma_fitted = gamma(a, 1, s)
    num_images = np.sum(schooling)*.9
    total = 0
    gamma_superSchooling1 = []
    while total < num_images:
        num_images_in_indiv_frag = gamma_fitted.rvs(1)
        total += num_images_in_indiv_frag
        gamma_superSchooling1.append(num_images_in_indiv_frag)
    min_gamma_superSchooling1, max_gamma_superSchooling1, mean_gamma_superSchooling1, std_gamma_superSchooling1 = np.min(gamma_superSchooling1), np.max(gamma_superSchooling1), np.mean(gamma_superSchooling1), np.std(gamma_superSchooling1)
    print("superSchooling1: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_superSchooling1, max_gamma_superSchooling1, mean_gamma_superSchooling1, std_gamma_superSchooling1))
    # hist_superSchooling1, bin_edges_superSchooling1 = np.histogram(gamma_superSchooling1, bins = 10 ** np.linspace(np.log10(min_gamma_superSchooling1), np.log10(max_gamma_superSchooling1), nbins))
    hist_superSchooling1, bin_edges_superSchooling1 = np.histogram(gamma_superSchooling1, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    ''' superSchooling2 '''
    a, s = gamma_params(85, 85)
    print("superSchooling2: a %.2f, s %.2f" %(a,s))
    gamma_fitted = gamma(1.5, 1, 50)
    # gamma_fitted = gamma(a, 1, s)
    num_images = np.sum(schooling)*.95
    total = 0
    gamma_superSchooling2 = []
    while total < num_images:
        num_images_in_indiv_frag = gamma_fitted.rvs(1)
        total += num_images_in_indiv_frag
        gamma_superSchooling2.append(num_images_in_indiv_frag)
    min_gamma_superSchooling2, max_gamma_superSchooling2, mean_gamma_superSchooling2, std_gamma_superSchooling2 = np.min(gamma_superSchooling2), np.max(gamma_superSchooling2), np.mean(gamma_superSchooling2), np.std(gamma_superSchooling2)
    print("superSchooling2: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_superSchooling2, max_gamma_superSchooling2, mean_gamma_superSchooling2, std_gamma_superSchooling2))
    # hist_superSchooling2, bin_edges_superSchooling2 = np.histogram(gamma_superSchooling2, bins = 10 ** np.linspace(np.log10(min_gamma_superSchooling2), np.log10(max_gamma_superSchooling2), nbins))
    hist_superSchooling2, bin_edges_superSchooling2 = np.histogram(gamma_superSchooling2, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    plt.ion()
    ''' schooling vs shoaling (POSTER) '''
    fig, ax = plt.subplots(1,1)

    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x*2/3/100,screen_y/100))
    labels = ['shoaling','schooling']
    colors = ['b', 'r']

    ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    ax.semilogx(bin_edges_schooling[:-1], hist_schooling, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)

    ax.legend(title="behavior type", fancybox=True)
    ax.set_xlabel('number of frames')
    ax.set_ylabel('number of individual fragments')

    fig.savefig('schooling_vs_shoaling.pdf', transparent=True)

    ''' simulated distributions (POSTER)'''
    fig, ax = plt.subplots(1,1)

    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x*2/3/100,screen_y/100))

    labels = ['shoaling','schooling',
                '%.2f' %np.mean(gamma_shoaling) + '    %.2f' %np.std(gamma_shoaling),
                '%.2f' %np.mean(gamma_schooling) + '    %.2f' %np.std(gamma_schooling),
                '%.2f' %np.mean(gamma_superSchooling1) + '      %.2f' %np.std(gamma_superSchooling1),
                '%.2f' %np.mean(gamma_superSchooling2) + '      %.2f' %np.std(gamma_superSchooling2)]
    # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    colors = ['b', 'r', 'g', 'y']

    ax.semilogx(bin_edges_gamma_shoaling[:-1], hist_gamma_shoaling, '--', markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    #ax.axvline(np.mean(gamma_shoaling), linestyle =  '--', color = colors[0])

    ax.semilogx(bin_edges_gamma_schooling[:-1], hist_gamma_schooling, '--', markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    #ax.axvline(np.mean(gamma_schooling), linestyle = '--', color = colors[1])

    ax.semilogx(bin_edges_superSchooling1[:-1], hist_superSchooling1, '--', markersize = 5, label = labels[4], color = colors[2], linewidth = 2)
    #ax.axvline(np.mean(gamma_superSchooling1), linestyle = '--', color = colors[2])

    ax.semilogx(bin_edges_superSchooling2[:-1], hist_superSchooling2, '--', markersize = 5, label = labels[5], color = colors[3], linewidth = 2)
    #ax.axvline(np.mean(gamma_superSchooling2), linestyle = '--', color = colors[3])

    ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    ax.semilogx(bin_edges_schooling[:-1], hist_schooling, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)

    ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    ax.set_xlabel('number of frames (nf)')
    ax.set_ylabel('number of individual fragments')

    # plt.show()
    fig.savefig('gamma_fitting_individual_fragments_distribution_computed.pdf', transparent=True)


    ''' fit with gamma, lognorm, weibull '''
    fig, ax = plt.subplots(1,1)

    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x*2/3/100,screen_y/100))

    labels = ['shoaling','schooling',
                '%.2f' %np.mean(gamma_shoaling) + '    %.2f' %np.std(gamma_shoaling)  + 'gamma',
                '%.2f' %np.mean(gamma_schooling) + '    %.2f' %np.std(gamma_schooling) + ' gamma',
                '%.2f' %np.mean(lognorm_shoaling) + '    %.2f' %np.std(lognorm_shoaling) + ' lognorm',
                '%.2f' %np.mean(lognorm_schooling) + '    %.2f' %np.std(lognorm_schooling) + ' lognorm',
                '%.2f' %np.mean(exponweib_shoaling) + '    %.2f' %np.std(exponweib_shoaling) + ' weibull',
                '%.2f' %np.mean(exponweib_schooling) + '    %.2f' %np.std(exponweib_schooling) + ' weibull']
    # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    colors = ['b', 'r', 'g', 'y']

    ax.semilogx(bin_edges_gamma_shoaling[:-1], hist_gamma_shoaling, '--', markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    #ax.axvline(np.mean(gamma_shoaling), linestyle =  '--', color = colors[0])

    ax.semilogx(bin_edges_gamma_schooling[:-1], hist_gamma_schooling, '--', markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    #ax.axvline(np.mean(gamma_schooling), linestyle = '--', color = colors[1])

    ax.semilogx(bin_edges_lognorm_shoaling[:-1], hist_lognorm_shoaling, '-.', markersize = 5, label = labels[4], color = colors[0], linewidth = 2)
    #ax.axvline(np.mean(lognorm_shoaling), linestyle =  '--', color = colors[0])

    ax.semilogx(bin_edges_lognorm_schooling[:-1], hist_lognorm_schooling, '-.', markersize = 5, label = labels[5], color = colors[1], linewidth = 2)
    #ax.axvline(np.mean(lognorm_schooling), linestyle = '--', color = colors[1])

    ax.semilogx(bin_edges_exponweib_shoaling[:-1], hist_exponweib_shoaling, ':', markersize = 5, label = labels[6], color = colors[0], linewidth = 2)
    #ax.axvline(np.mean(exponweib_shoaling), linestyle =  '--', color = colors[0])

    ax.semilogx(bin_edges_exponweib_schooling[:-1], hist_exponweib_schooling, ':', markersize = 5, label = labels[7], color = colors[1], linewidth = 2)
    #ax.axvline(np.mean(exponweib_schooling), linestyle = '--', color = colors[1])

    ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    ax.semilogx(bin_edges_schooling[:-1], hist_schooling, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)

    ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    ax.set_xlabel('number of frames (nf)')
    ax.set_ylabel('number of individual fragments')

    # ''' fit with lognorm '''
    # fig, ax = plt.subplots(1,1)
    #
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x*2/3/100,screen_y/100))
    #
    # labels = ['shoaling','schooling',
    #             '%.2f' %np.mean(lognorm_shoaling) + '    %.2f' %np.std(lognorm_shoaling),
    #             '%.2f' %np.mean(lognorm_schooling) + '    %.2f' %np.std(lognorm_schooling)]
    # # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    # colors = ['b', 'r', 'g', 'y']
    #
    # ax.semilogx(bin_edges_lognorm_shoaling[:-1], hist_lognorm_shoaling, '--', markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    # #ax.axvline(np.mean(lognorm_shoaling), linestyle =  '--', color = colors[0])
    #
    # ax.semilogx(bin_edges_lognorm_schooling[:-1], hist_lognorm_schooling, '--', markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    # #ax.axvline(np.mean(lognorm_schooling), linestyle = '--', color = colors[1])
    #
    # ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    # ax.semilogx(bin_edges_schooling[:-1], hist_schooling, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)
    #
    # ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    # ax.set_xlabel('number of frames (nf)')
    # ax.set_ylabel('number of individual fragments')
    #
    # ''' fit with weibull '''
    # fig, ax = plt.subplots(1,1)
    #
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x*2/3/100,screen_y/100))
    #
    # labels = ['shoaling','schooling',
    #             '%.2f' %np.mean(exponweib_shoaling) + '    %.2f' %np.std(exponweib_shoaling),
    #             '%.2f' %np.mean(exponweib_schooling) + '    %.2f' %np.std(exponweib_schooling)]
    # # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    # colors = ['b', 'r', 'g', 'y']
    #
    # ax.semilogx(bin_edges_exponweib_shoaling[:-1], hist_exponweib_shoaling, '--', markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    # #ax.axvline(np.mean(exponweib_shoaling), linestyle =  '--', color = colors[0])
    #
    # ax.semilogx(bin_edges_exponweib_schooling[:-1], hist_exponweib_schooling, '--', markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    # #ax.axvline(np.mean(exponweib_schooling), linestyle = '--', color = colors[1])
    #
    # ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    # ax.semilogx(bin_edges_schooling[:-1], hist_schooling, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)
    #
    # ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    # ax.set_xlabel('number of frames (nf)')
    # ax.set_ylabel('number of individual fragments')
