from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.stats import gamma
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
#import matplotlib
#font = {'family' : 'normal',
#        'size'   : 18}
#matplotlib.rc('font', **font)

def gamma_params(mean, std):
    var = std**2
    a = (mean/var)**2
    s = var**2/mean
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

    print("Loading data ...")

    shoaling = np.load('shoaling_individual_fragments_distribution.npy')
    shoaling =  filter(lambda x: x >= 1, shoaling)
    min_shoaling, max_shoaling, mean_shoaling, std_shoaling = np.min(shoaling), np.max(shoaling), np.mean(shoaling), np.std(shoaling)
    print("Shoaling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_shoaling, max_shoaling, mean_shoaling, std_shoaling))

    schooling = np.load('schooling_individual_fragments_distribution.npy')
    schooling =  filter(lambda x: x >= 1, schooling)
    min_schooling, max_schooling, mean_schooling, std_schooling = np.min(schooling), np.max(schooling), np.mean(schooling), np.std(schooling)
    print("Shoaling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_schooling, max_schooling, mean_schooling, std_schooling))

    plt.ion()
    fig, ax_arr = plt.subplots(2,3)
    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x/100,screen_y/100))
    nbins = 20
    colors = ['b', 'r', 'g', 'y']
    ''' distribution of number of frames per individual fragment '''
    hist_shoaling_real, bin_edges_shoaling_real = np.histogram(shoaling, bins = nbins,  normed = True)
    hist_schooling_real, bin_edges_schooling_real = np.histogram(schooling, bins = nbins, normed = True)
    ax = ax_arr[0,0]
    ax.plot(bin_edges_shoaling_real[:-1], hist_shoaling_real, color = colors[0], linewidth = 2)
    ax.plot(bin_edges_schooling_real[:-1], hist_schooling_real, color = colors[1], linewidth = 2)
    ax.set_xlabel('number of frames in individual fragment')
    ax.set_ylabel('Probability')

    ''' distribution of the logarithm of the number of frames per individual fragment'''
    hist_shoaling_real, bin_edges_shoaling_real = np.histogram(np.log10(shoaling), bins = nbins,  normed = True)
    hist_schooling_real, bin_edges_schooling_real = np.histogram(np.log10(schooling), bins = nbins, normed = True)
    ax = ax_arr[1,0]
    ax.plot(bin_edges_shoaling_real[:-1], hist_shoaling_real, color = colors[0], linewidth = 2)
    ax.plot(bin_edges_schooling_real[:-1], hist_schooling_real, color = colors[1], linewidth = 2)
    ax.set_xlabel('log10(number of frames in individual fragment)')
    ax.set_ylabel('Probability')

    ''' fit of the data to a gamma distribution '''
    a_shoaling, loc_shoaling, scale_shoaling = gamma.fit(shoaling, loc = 0.99)
    print("Gamma fitting to shoaling: a %.2f, loc %.2f, scale %.2f" %(a_shoaling, loc_shoaling, scale_shoaling))
    fitted_mean_shoaling, fitted_std_shoaling = gamma.stats(a_shoaling, loc = loc_shoaling, scale = scale_shoaling, moments='mv')
    print('mean %.2f, std %.2f' %(fitted_mean_shoaling, np.sqrt(fitted_std_shoaling)))
    ks_stat, p_value = stats.kstest(shoaling, 'gamma', args = (a_shoaling, loc_shoaling, scale_shoaling), N = len(shoaling))
    print('Goodness of fit: %f %f' %(ks_stat, p_value))
    gamma_shoaling = gamma(a_shoaling, loc_shoaling, scale_shoaling)

    a_schooling, loc_schooling, scale_schooling = gamma.fit(schooling, loc = 0.99)
    print("Gamma fitting to schooling: a %.2f, loc %.2f, scale %.2f" %(a_schooling, loc_schooling, scale_schooling))
    fitted_mean_schooling, fitted_std_schooling = gamma.stats(a_schooling, loc = loc_schooling, scale = scale_schooling, moments='mv')
    print('mean %.2f, std %.2f' %(fitted_mean_schooling, np.sqrt(fitted_std_schooling)))
    ks_stat, p_value = stats.kstest(schooling, 'gamma', args = (a_schooling, loc_schooling, scale_schooling), N = len(schooling))
    print('Goodness of fit: %f %f' %(ks_stat, p_value))
    gamma_schooling = gamma(a_schooling, loc_schooling, scale_schooling)

    ax = ax_arr[0,1]
    x = np.linspace(min_shoaling, max_shoaling, 100)
    ax.plot(x, gamma_shoaling.pdf(x), color = colors[0])
    x = np.linspace(min_schooling, max_schooling, 100)
    ax.plot(x, gamma_schooling.pdf(x), color = colors[1])
    ax.set_ylim(ax_arr[0,0].get_ylim())
    ax.set_xlim(ax_arr[0,0].get_xlim())
    ax.set_xlabel('number of frames in individual fragment')

    ax = ax_arr[1,1]
    x = 10**np.linspace(np.log10(min_shoaling), np.log10(max_shoaling), 100)
    ax.plot(np.log10(x), gamma_shoaling.pdf(x), color = colors[0])
    x = 10**np.linspace(np.log10(min_schooling), np.log10(max_schooling), 100)
    ax.plot(np.log10(x), gamma_schooling.pdf(x), color = colors[1])
    ax.set_xlim(ax_arr[1,0].get_xlim())
    ax.set_ylim(ax_arr[1,0].get_ylim())
    ax.set_xlabel('log10(number of frames in individual fragment)')

    ''' generating values from distributions '''
    gamma_shoaling_values = gamma_shoaling.rvs(10000000)
    min_gamma_shoaling, max_gamma_shoaling, mean_gamma_shoaling, std_gamma_shoaling = np.min(gamma_shoaling_values), \
                                                                                        np.max(gamma_shoaling_values), \
                                                                                        np.mean(gamma_shoaling_values), \
                                                                                        np.std(gamma_shoaling_values)
    print("Shoaling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_shoaling, max_gamma_shoaling, mean_gamma_shoaling, std_gamma_shoaling))

    gamma_schooling_values = gamma_schooling.rvs(1000000)
    min_gamma_schooling, max_gamma_schooling, mean_gamma_schooling, std_gamma_schooling = np.min(gamma_schooling_values), \
                                                                                            np.max(gamma_schooling_values), \
                                                                                            np.mean(gamma_schooling_values), \
                                                                                            np.std(gamma_schooling_values)
    print("Shoaling: min %.2f, max %.2f, mean %.2f, std %.2f" %(min_gamma_schooling, max_gamma_schooling, mean_gamma_schooling, std_gamma_schooling))

    ''' distribution of number of frames per individual fragment '''
    hist_gamma_shoaling_real, bin_edges_gamma_shoaling_real = np.histogram(gamma_shoaling_values, bins = nbins,  normed = True)
    hist_schooling_real, bin_edges_schooling_real = np.histogram(schooling, bins = nbins, normed = True)
    ax = ax_arr[0,2]
    ax.plot(bin_edges_gamma_shoaling_real[:-1], hist_gamma_shoaling_real, color = colors[0], linewidth = 2)
    ax.plot(bin_edges_schooling_real[:-1], hist_schooling_real, color = colors[1], linewidth = 2)
    ax.set_xlabel('number of frames in individual fragment')

    ''' distribution of the logarithm of the number of frames per individual fragment'''
    hist_gamma_shoaling_real, bin_edges_gamma_shoaling_real = np.histogram(np.log10(gamma_shoaling_values), bins = nbins,  normed = True)
    hist_schooling_real, bin_edges_schooling_real = np.histogram(np.log10(schooling), bins = nbins, normed = True)
    ax = ax_arr[1,2]
    ax.plot(bin_edges_gamma_shoaling_real[:-1], hist_gamma_shoaling_real, color = colors[0], linewidth = 2)
    ax.plot(bin_edges_schooling_real[:-1], hist_schooling_real, color = colors[1], linewidth = 2)
    ax.set_xlabel('log10(number of frames in individual fragment)')

    ''' Understanding the relationship of the gamma distribution parameters with mean a variance'''
    n = 10
    a = np.tile(np.linspace(0,10,n),(n,1))
    s = np.tile(np.linspace(0,10000,n),(n,1)).T
    mean, std = mean_std_gamma(a,s)

    fig, ax_arr = plt.subplots(2,2)
    ax = ax_arr[0,0]
    im = ax.imshow(np.log10(mean))
    ax.set_ylabel('scale (s)')
    ax.set_xlabel('shape (a)')
    ax.set_title('(log10mean)')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    ax = ax_arr[0,1]
    im = ax.imshow(std)
    ax.set_ylabel('scale (s)')
    ax.set_xlabel('shape (a)')
    ax.set_title('std')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    std = np.tile(np.linspace(20,500,n),(n,1))
    mean = np.tile(10**np.linspace(np.log10(1),np.log10(1000),n),(n,1)).T
    a, s  = gamma_params(mean, std)

    ax = ax_arr[1,0]
    im = ax.imshow(a)
    ax.set_ylabel('std')
    ax.set_xlabel('mean')
    ax.set_title('a (shape)')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)

    ax = ax_arr[1,1]
    im = ax.imshow(std)
    ax.set_ylabel('var')
    ax.set_xlabel('mean')
    ax.set_title('s (scale)')
    ax.invert_yaxis()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)


    plt.show()

    # MIN = np.min([np.min(shoaling),np.min(schooling)])
    # MIN = 1
    # MAX = np.max([np.max(shoaling),np.max(schooling)])
    # #x = np.linspace(MIN,MAX,1000)
    # x = np.linspace(MIN, MAX, 1000)
    #
    # ''' shoaling '''
    # a, loc, scale = gamma.fit(shoaling,loc=1)
    # gamma_fitted = gamma(a, loc, scale)
    # gamma_shoaling = gamma_fitted.rvs(1000000)
    # y_shoaling = gamma_fitted.pdf(x)
    #
    # ''' schooling '''
    # a, loc, scale= gamma.fit(schooling,loc=1)
    # gamma_fitted = gamma(a, loc, scale)
    # gamma_schooling = gamma_fitted.rvs(1000000)
    # y_schooling = gamma_fitted.pdf(x)
    #
    # ''' wild super-schooling '''
    # a,s = gamma_params(75, 100)
    # print(a,s)
    # gamma_fitted = gamma(a, 1, s)
    # gamma_superSchooling1 = gamma_fitted.rvs(1000000)
    # y_superSchooling1 = gamma_fitted.pdf(x)
    #
    # a,s = gamma_params(75, 50)
    # print(a,s)
    # gamma_fitted = gamma(0.35, 1, 250)
    # gamma_superSchooling2 = gamma_fitted.rvs(1000000)
    # y_superSchooling2 = gamma_fitted.pdf(x)
    #
    # # gamma_fitted = gamma(0.2, 1, 321)
    # # gamma_superSchooling2 = gamma_fitted.rvs(len(schooling))
    #
    # plt.ion()
    # ''' schooling vs shoaling '''
    # # number of frames in individual fragments
    # nbins = 30
    # hist_shoaling, bin_edges_shoaling = np.histogram(gamma_shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    # hist_schooling, bin_edges_schooling = np.histogram(gamma_schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    # hist_shoaling_real, bin_edges_shoaling_real = np.histogram(shoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    # hist_schooling_real, bin_edges_schooling_real = np.histogram(schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    # hist_superSchooling1, bin_edges_superSchooling1 = np.histogram(gamma_superSchooling1, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    # hist_superSchooling2, bin_edges_superSchooling2 = np.histogram(gamma_superSchooling2, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins), normed = True)
    #
    # fig, ax = plt.subplots(1,1)
    #
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x/100,screen_y/100))
    # labels = ['shoaling','schooling']
    # colors = ['b', 'r']
    #
    # ax.semilogx(bin_edges_shoaling_real[:-1], hist_shoaling_real, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    # ax.semilogx(bin_edges_schooling_real[:-1], hist_schooling_real, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)
    #
    # ax.legend(title="behavior type", fancybox=True)
    # ax.set_xlabel('number of frames')
    # ax.set_ylabel('number of individual fragments')
    #
    # fig.savefig('schooling_vs_shoaling.pdf', transparent=True)
    #
    # ''' simulated distributions '''
    # fig, ax = plt.subplots(1,1)
    #
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x/100,screen_y/100))
    #
    # labels = ['shoaling','schooling',
    #             '%.2f' %np.mean(gamma_shoaling) + '    %.2f' %np.std(gamma_shoaling),
    #             '%.2f' %np.mean(gamma_schooling) + '    %.2f' %np.std(gamma_schooling),
    #             '%.2f' %np.mean(gamma_superSchooling1) + '      %.2f' %np.std(gamma_superSchooling1),
    #             '%.2f' %np.mean(gamma_superSchooling2) + '      %.2f' %np.std(gamma_superSchooling2)]
    # # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    # colors = ['b', 'r', 'g', 'y']
    #
    # ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling, '--', markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    # #ax.axvline(np.mean(gamma_shoaling), linestyle =  '--', color = colors[0])
    #
    # ax.semilogx(bin_edges_schooling[:-1], hist_schooling, '--', markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    # #ax.axvline(np.mean(gamma_schooling), linestyle = '--', color = colors[1])
    #
    # ax.semilogx(bin_edges_superSchooling1[:-1], hist_superSchooling1, '--', markersize = 5, label = labels[4], color = colors[2], linewidth = 2)
    # #ax.axvline(np.mean(gamma_superSchooling1), linestyle = '--', color = colors[2])
    #
    # ax.semilogx(bin_edges_superSchooling2[:-1], hist_superSchooling2, '--', markersize = 5, label = labels[5], color = colors[3], linewidth = 2)
    # #ax.axvline(np.mean(gamma_superSchooling2), linestyle = '--', color = colors[3])
    #
    # ax.semilogx(bin_edges_shoaling_real[:-1], hist_shoaling_real, markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    # ax.semilogx(bin_edges_schooling_real[:-1], hist_schooling_real, markersize = 5, label = labels[1], color = colors[1], linewidth = 2)
    #
    # ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    # ax.set_xlabel('number of frames (nf)')
    # ax.set_ylabel('number of individual fragments')
    #
    # ''' theoretical distributions '''
    # fig, ax = plt.subplots(1,1)
    # window = plt.get_current_fig_manager().window
    # screen_y = window.winfo_screenheight()
    # screen_x = window.winfo_screenwidth()
    # fig.set_size_inches((screen_x/100,screen_y/100))
    #
    # labels = ['shoaling','schooling',
    #             '%.2f' %np.mean(gamma_shoaling) + '    %.2f' %np.std(gamma_shoaling),
    #             '%.2f' %np.mean(gamma_schooling) + '    %.2f' %np.std(gamma_schooling),
    #             '%.2f' %np.mean(gamma_superSchooling1) + '      %.2f' %np.std(gamma_superSchooling1),
    #             '%.2f' %np.mean(gamma_superSchooling2) + '      %.2f' %np.std(gamma_superSchooling2)]
    # # labels = ['shoaling', 'schooling', 'shoaling-fitted', 'schooling-fitted', 'superSchooling']
    # colors = ['b', 'r', 'g', 'y']
    #
    # ax.plot(x,y_shoaling,color = colors[0])
    # ax.plot(x,y_schooling,color = colors[1])
    # ax.plot(x,y_superSchooling1,color = colors[2])
    # ax.plot(x,y_superSchooling2,color = colors[3])
    #
    # ax.legend(title="        E[nf]      Var[nf]", fancybox=True)
    # ax.set_xlabel('number of frames (nf)')
    # ax.set_ylabel('number of individual fragments')
    #
    # # plt.show()
    # fig.savefig('gamma_fitting_individual_fragments_distribution.pdf', transparent=True)
