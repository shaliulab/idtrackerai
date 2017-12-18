from scipy.stats import gamma
import numpy as np
from matplotlib import pyplot as plt

def get_next_number_of_blobs_in_fragment(gamma_sim, min_number_of_frames_per_fragment, max_number_of_frames_per_fragment):
    number_of_frames_per_fragment = int(np.floor(gamma_sim.rvs(1)))
    while number_of_frames_per_fragment < min_number_of_frames_per_fragment or number_of_frames_per_fragment > max_number_of_frames_per_fragment:
        number_of_frames_per_fragment = int(np.floor(gamma_sim.rvs(1)))

    return number_of_frames_per_fragment

def pdf2logpdf(pdf):
    def logpdf(x):
        return pdf(x)*x*np.log(10)
    return logpdf

if __name__ == '__main__':
    scale_parameters = [2000, 1000, 500, 250, 100][::-1]
    shape_parameters = [0.5, 0.35, 0.25, 0.15, 0.05][::-1]
    loc_parameter = 0.99
    min_number_of_frames_per_fragment = 1
    max_number_of_frames_per_fragment = 10000
    number_of_fragments = 1000
    MIN = 1
    MAX = max_number_of_frames_per_fragment

    plt.ion()
    fig, ax_arr = plt.subplots(len(shape_parameters), len(scale_parameters), sharex = True, sharey = False)
    for column, scale_parameter in enumerate(scale_parameters):
        for row, shape_parameter in enumerate(shape_parameters):
            print("Gamma parameters from which to draw the values: ", shape_parameter, scale_parameter, loc_parameter)
            gamma_sim = gamma(a = shape_parameter, loc = loc_parameter, scale = scale_parameter)
            # Pdf of theoretical gamma
            gamma_logpdf = pdf2logpdf(gamma_sim.pdf)
            # Get fragments length (integers) from 1 to 10000. We are truncating
            number_of_images_in_individual_fragments_truncated = []
            for i in range(number_of_fragments):
                n = get_next_number_of_blobs_in_fragment(gamma_sim, min_number_of_frames_per_fragment, max_number_of_frames_per_fragment)
                number_of_images_in_individual_fragments_truncated.append(n)
            # fit new gamma to truncated values
            shape, loc, scale = gamma.fit(number_of_images_in_individual_fragments_truncated, floc = loc_parameter)
            print("Gamma parameters of the fit to the truncated values: ", shape, scale, loc)
            gamma_fitted = gamma(shape, loc, scale)
            gamma_fitted_logpdf = pdf2logpdf(gamma_fitted.pdf)
            # plot
            nbins = 10
            ax = ax_arr[row, column]

            logbins = np.linspace(np.log10(MIN), np.log10(MAX), nbins)
            n, _, _ = ax.hist(np.log10(number_of_images_in_individual_fragments_truncated), bins = logbins, normed = True, label = 'truncated')
            logbins2 = np.linspace(np.log10(MIN), np.log10(MAX), 100)
            ax.plot(logbins2, gamma_logpdf(np.power(10,logbins2)), 'r', label = 'original')
            ax.plot(logbins2, gamma_fitted_logpdf(np.power(10,logbins2)), 'b', label = 'fitted')
            ax.set_xlim((np.log10(MIN), np.log10(MAX)))

            title = 'a_fit = %.2f, s_fit = %.2f' %(shape, scale)
            ax.text(2.25, 1.5, title, horizontalalignment = 'center')

            ax.set_ylim((0, 2))
            if row == len(shape_parameters)-1:
                ax.set_xlabel('number of frames \n\nscale = %.2f' %scale_parameter)
                ax.set_xticklabels([1,10,100,1000,10000])
            else:
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel('shape = %.2f \n\nPDF' %shape_parameter)
            else:
                ax.set_yticklabels([])

    plt.show()
