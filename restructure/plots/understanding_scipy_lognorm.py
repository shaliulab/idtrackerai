import numpy as np
from scipy.stats import lognorm, gamma
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

if __name__ == '__main__':

    ''' schoaling '''
    schoaling = np.load('library/shoaling_individual_fragments_distribution.npy')
    schoaling =  filter(lambda x: x >= 1, schoaling)
    a_gamma, loc_gamma, scale_gamma = gamma.fit(schoaling,loc=1)
    print("schoaling: ", a_gamma, loc_gamma, scale_gamma)
    gamma_fitted = gamma(a_gamma, loc_gamma, scale_gamma)
    gamma_schoaling = gamma_fitted.rvs(len(schoaling))

    ''' schooling '''
    schooling = np.load('library/schooling_individual_fragments_distribution.npy')
    schooling =  filter(lambda x: x >= 1, schooling)
    a_gamma, loc_gamma, scale_gamma = gamma.fit(schooling,loc=1)
    print("schooling: ", a_gamma, loc_gamma, scale_gamma)
    gamma_fitted = gamma(a_gamma, loc_gamma, scale_gamma)
    gamma_schooling = gamma_fitted.rvs(len(schooling))

    ''' wild super-schooling '''
    gamma_fitted = gamma(1.5, 1, 50)
    gamma_superSchooling1 = gamma_fitted.rvs(len(schooling))

    gamma_fitted = gamma(2, 1, 10)
    gamma_superSchooling2 = gamma_fitted.rvs(len(schooling))

    plt.ion()
    fig, ax = plt.subplots(1,1)

    window = plt.get_current_fig_manager().window
    screen_y = window.winfo_screenheight()
    screen_x = window.winfo_screenwidth()
    fig.set_size_inches((screen_x/100,screen_y/100))

    # number of frames in individual fragments
    nbins = 30
    MIN = np.min([np.min(gamma_schoaling),np.min(gamma_schooling),np.min(gamma_superSchooling1),np.min(gamma_superSchooling2)])
    MAX = np.max([np.max(gamma_schoaling),np.max(gamma_schooling),np.max(gamma_superSchooling1),np.min(gamma_superSchooling2)])
    print(MIN, MAX)

    hist_shoaling, bin_edges_shoaling = np.histogram(gamma_schoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    hist_schooling, bin_edges_schooling = np.histogram(gamma_schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    hist_shoaling_real, bin_edges_shoaling_real = np.histogram(schoaling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    hist_schooling_real, bin_edges_schooling_real = np.histogram(schooling, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    hist_superSchooling1, bin_edges_superSchooling1 = np.histogram(gamma_superSchooling1, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))
    hist_superSchooling2, bin_edges_superSchooling2 = np.histogram(gamma_superSchooling2, bins = 10 ** np.linspace(np.log10(MIN), np.log10(MAX), nbins))

    labels = ['real-schoaling','real-shcooling','fitted-schoaling','fitted-schooling','super-schooling1','super_schooling2']
    colors = ['b', 'r', 'g', 'y']

    ax.semilogx(bin_edges_shoaling_real[:-1], hist_shoaling_real , '--', markersize = 5, label = labels[0], color = colors[0], linewidth = 2)
    ax.semilogx(bin_edges_shoaling[:-1], hist_shoaling ,markersize = 5, label = labels[2], color = colors[0], linewidth = 2)
    ax.axvline(np.mean(gamma_schoaling), linestyle =  '-', color = colors[0])

    ax.semilogx(bin_edges_schooling_real[:-1], hist_schooling_real , '--', markersize = 5, label = labels[1], color = colors[1], linewidth = 2)
    ax.semilogx(bin_edges_schooling[:-1], hist_schooling ,markersize = 5, label = labels[3], color = colors[1], linewidth = 2)
    ax.axvline(np.mean(gamma_schooling), linestyle = '-', color = colors[1])

    ax.semilogx(bin_edges_superSchooling1[:-1], hist_superSchooling1 ,markersize = 5, label = labels[4], color = colors[2], linewidth = 2)
    ax.axvline(np.mean(gamma_superSchooling1), linestyle = '-', color = colors[2])

    ax.semilogx(bin_edges_superSchooling2[:-1], hist_superSchooling2 ,markersize = 5, label = labels[5], color = colors[3], linewidth = 2)
    ax.axvline(np.mean(gamma_superSchooling2), linestyle = '-', color = colors[3])

    ax.legend(title="behavior type", fancybox=True)
    ax.set_xlabel('number of frames')
    ax.set_ylabel('number of individual fragments')

    # plt.show()
    fig.savefig('gamma_fitting_individual_fragments_distribution.pdf', transparent=True)
