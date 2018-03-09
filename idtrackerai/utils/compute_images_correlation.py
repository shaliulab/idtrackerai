# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., De Polavieja, G.G.,
# (2018). idtracker.ai: Tracking all individuals with correct identities in large
# animal collectives (submitted)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import sys
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
from idtrackerai.video import Video
from idtrackerai.visualise_cnn import visualise
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.network.identification_model.network_params import NetworkParams

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def compare_two_images(image1, image2):
    return 1 - (ssim(image1, image2, data_range = image2.max() - image2.min()) + 1) / 2

def compare_all_images(images):
    distances = []

    for image1, image2 in combinations(images,2):
        distances.append(compare_two_images(image1, image2))

    return distances

if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/video_object.npy').item()
    list_of_global_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/preprocessing/global_fragments.npy').item()
    list_of_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/preprocessing/fragments.npy').item()
    list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
    first_global_fragment = list_of_global_fragments.global_fragments[0]
    images = first_global_fragment.individual_fragments[0].images
    images = np.asarray(images)
    number_of_images = 1200
    distances = compare_all_images(images[:number_of_images])
    Z = linkage(distances, 'ward')
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(111)
    ax.set_title('Hierarchical Clustering Dendrogram')
    ax.set_xlabel('sample index')
    ax.set_ylabel('distance')
    R = dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    ax.set_ylim([-0.1, ax.get_ylim()[1]])
    positions = ax.get_xticks()
    labels = ax.get_xticklabels()

    for i in range(number_of_images):
        label = labels[i].get_text()
        imscatter(positions[i], 0, images[int(label)], ax=None, zoom=.2)
    ax.grid(False)
    plt.show()

    C = fcluster(Z, 2, criterion = 'maxclust')
    im1 = np.median(images[np.where(C == 1)[0]], axis = 0)

    plt.imshow(im1)
    plt.show()
    params = NetworkParams(video.number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = None,
                                save_folder = video.accumulation_folder,
                                restore_folder = video.accumulation_folder,
                                image_size = video.identification_image_size,
                                video_path = video.video_path)
    net = ConvNetwork(params, training_flag = False)
    net.restore()
    im1 = np.expand_dims(im1, 3)
    visualise(video,net, [im1], None)
