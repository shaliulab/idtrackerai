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
# (2018). idtracker.ai: Tracking all individuals in large collectives of unmarked animals (R-F.,F. and B.,M. contributed equally to this work.)


from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import os
import sys
from idtrackerai.tf_cnnvisualisation.tf_cnnvis import *
from idtrackerai.network.identification_model.id_CNN import ConvNetwork
from idtrackerai.network.identification_model.network_params import NetworkParams
from idtrackerai.video import Video
from idtrackerai.globalfragment import GlobalFragment
if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.visualise_cnn")

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.int16)
    indices = (index_offset + labels.ravel()).astype('int')
    labels_one_hot.flat[indices] = 1
    return labels_one_hot

def visualise(video_object, net, images, labels = None):
    video_object.cnn_visualiser_logs_folder = os.path.join(video_object._session_folder, 'cnn_visualiser_logs')
    logger.debug("logs folder :%s" %video_object.cnn_visualiser_logs_folder)
    video_object.cnn_visualiser_images_folder = os.path.join(video_object._session_folder, 'cnn_visualiser_images')
    logger.debug("images folder :%s" %video_object.cnn_visualiser_images_folder)
    if not os.path.isdir(video_object.cnn_visualiser_logs_folder):
        os.makedirs(video_object.cnn_visualiser_logs_folder)
        logger.debug("logs folder created")
    if not os.path.isdir(video_object.cnn_visualiser_images_folder):
        os.makedirs(video_object.cnn_visualiser_images_folder)
        logger.debug("images folder created")
    net.training_flag = False
    #restore variables from the pretraining
    logger.debug("restoring network from accumulation")
    # deconv visualization
    # images = tf.placeholder(tf.float32, [None, video_object.identification_image_size[0], video_object.identification_image_size[1], video_object.identification_image_size[2]], name = 'images')
    # y_ = tf.placeholder(tf.float32, [None, video_object.number_of_animals])

    layers = ["p", "r"] #r : output relu layers, p : output pooling layers, c : output convolutional layers
    logger.debug("Start deconvolution")
    if labels is not None:
        labels = dense_to_one_hot(labels, video_object.number_of_animals)


    # api call
    logger.debug("Logs folder: %s" %video_object.cnn_visualiser_logs_folder)
    logger.debug("Images folder: %s" %video_object.cnn_visualiser_images_folder)
    logger.debug("Default graph: %s" %tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {net.x_pl: images}
        logger.debug("feed dict: %s" %feed_dict)


    is_success = deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = feed_dict,
                                      input_tensor = net.x_pl, layers = layers, path_logdir = video_object.cnn_visualiser_logs_folder,
                                      path_outdir = video_object.cnn_visualiser_images_folder)

    is_success = activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = feed_dict,
                                      input_tensor = net.x_pl, layers = layers, path_logdir = video_object.cnn_visualiser_logs_folder,
                                      path_outdir = video_object.cnn_visualiser_images_folder)

    logger.debug("Done")
    video_object.save()

if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/video_object.npy', allow_pickle=True).item()
    list_of_global_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/preprocessing/global_fragments.npy', allow_pickle=True).item()
    list_of_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/Cafeina5pecesLarge/session_20171214/preprocessing/fragments.npy', allow_pickle=True).item()
    list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
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
    first_global_fragment = list_of_global_fragments.global_fragments[0]
    images = []
    labels = []
    number_of_images_per_individual = 1

    for i in range(video.number_of_animals):
        images.extend(first_global_fragment.individual_fragments[i].images[:number_of_images_per_individual])
        labels.extend([first_global_fragment.individual_fragments[i].final_identity -1] * number_of_images_per_individual)

    images = np.expand_dims(np.asarray(images), 3)
    labels = np.expand_dims(np.asarray(labels),1)
    visualise(video,net, images, None)
