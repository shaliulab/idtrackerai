from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import matplotlib
#font = {'family' : 'normal',
#       'size'   : 18}
#matplotlib.rc('font', **font)

class crossing_intervals():
    def __init__(self, crossings):
        self.crossings = crossings[np.where(crossings>0)]
    @classmethod
    def load_from(cls, crossing_file):
        cls(np.load(crossings_file))

def produce_histogram(data, log=False, nbins = 30):
    if log:
        h, bin_edges = np.histogram(data, bins = np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), nbins))
    else:
        h, bin_edges = np.histogram(data, bins = np.linspace(np.min(data), np.max(data), nbins))
    return h, bin_edges

def plot_histogram(data, ax, nbins = 30):
    bins = np.linspace(np.min(data), np.max(data), nbins)
    return ax.hist(data, bins=bins, normed = True)

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

def loglikelihood(distribution,x):
    return np.sum(distribution.logpdf(x))

def resample_and_remove_first(bins, n=100):
    resampled = np.linspace(bins[0], bins[-1], n)
    return resampled[1:]

def histogram_plus_fits(data, ax, axlog, distributions = [], colours = []):
    _, bins, _ = plot_histogram(data, ax)
    _, logbins, _ = plot_histogram(np.log10(data), axlog)

    bins = resample_and_remove_first(bins)
    logbins = resample_and_remove_first(logbins)

    for i, distribution in enumerate(distributions):
        if distribution.numargs>0:
            *a, loc, scale = distribution.fit(data, floc=0)
            print(distribution.name + " parameters:", a, loc, scale)
            fitted_distribution = distribution(*a, loc = loc, scale = scale)

        else:
            loc, scale = distribution.fit(data, floc=0)
            print(distribution.name + " parameters:", loc, scale)
            fitted_distribution = distribution(loc = loc, scale = scale)

        print(distribution.name + " loglikelihood:", loglikelihood(fitted_distribution, data))
        fitted_distribution_logpdf = pdf2logpdf(fitted_distribution.pdf)
        ax.plot(bins, fitted_distribution.pdf(bins), colours[i])
        axlog.plot(logbins, fitted_distribution_logpdf(np.power(10,logbins)), colours[i])

if __name__ == '__main__':
    f, ((ax1, ax2), (ax1log, ax2log)) = plt.subplots(2,2)
    distributions = [ss.expon, ss.gamma, ss.lognorm, ss.exponweib, ss.fatiguelife, ss.chi2]
    colours = ['k', 'r', 'g', 'b', 'y', 'm']

    print("Loading shoaling...")
    shoaling = np.load('shoaling_individual_fragments_distribution.npy')
    shoaling = shoaling[np.where(shoaling>0)]
    histogram_plus_fits(shoaling, ax1, ax1log, distributions = distributions, colours = colours)

    print("Loading schooling")
    schooling = np.load('schooling_individual_fragments_distribution.npy')
    schooling = schooling[np.where(schooling>0)]
    histogram_plus_fits(schooling, ax2, ax2log, distributions = distributions, colours = colours)



    plt.show()
