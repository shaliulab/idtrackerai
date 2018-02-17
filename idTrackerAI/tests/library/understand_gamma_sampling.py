from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    a_list = [0.25, 0.5, .75, 1, 1.5]
    s_list = [50, 100, 250, 500, 800]
    mean_list = []
    sigma_list = []
    for a in a_list:
        for s in s_list:
            mean_list.append(a * s)
            sigma_list.append(np.sqrt(a * s**2))

    plt.ion()
    plt.figure()
    plt.plot(mean_list, sigma_list, 'o')
    plt.plot([np.min(mean_list), np.max(mean_list)], [np.min(mean_list), np.max(mean_list)], 'k-')
    plt.xlabel('mean')
    plt.ylabel('sigma')
    plt.show()
