from __future__ import absolute_import, division, print_function
import time
import numpy as np
import tensorflow as tf
import os
import logging
import sys
sys.path.append('./utils')
sys.path.append('./network')
sys.path.append('./network/identification_model')
sys.path.append('./tf_cnnvisualisation')

from tf_cnnvis import *
from video import Video
from globalfragment import GlobalFragment
from id_CNN import ConvNetwork
from network_params import NetworkParams

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

def visualise(video_object, net, image, label):
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

    layers = ["r", "p", "c"] #r : output relu layers, p : output pooling layers, c : output convolutional layers
    logger.debug("Start deconvolution")
    label = dense_to_one_hot([label], video_object.number_of_animals)


    # api call
    logger.debug("Logs folder: %s" %video_object.cnn_visualiser_logs_folder)
    logger.debug("Images folder: %s" %video_object.cnn_visualiser_images_folder)
    logger.debug("Default graph: %s" %tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {net.x_pl: image, net.y_target_pl: label}
        logger.debug("feed dict: %s" %feed_dict)


    is_success = deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = feed_dict,
                                      input_tensor = net.x_pl, layers = layers, path_logdir = video_object.cnn_visualiser_logs_folder,
                                      path_outdir = video_object.cnn_visualiser_images_folder)

    is_success = activation_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = feed_dict,
                                      input_tensor = net.x_pl, layers = layers, path_logdir = video_object.cnn_visualiser_logs_folder,
                                      path_outdir = video_object.cnn_visualiser_images_folder)
    # layer = 'saver_fc_softmax/Assign'
    # feed_dict = {net.x_pl : image}
    # is_success = deepdream_visualization(graph_or_path = tf.get_default_graph(),
    #                                     value_feed_dict = feed_dict,
    #                                     layer=layer,
    #                                     classes = range(1, video_object.number_of_animals + 1),
    #                                     path_logdir = video_object.cnn_visualiser_logs_folder,
    #                                     path_outdir = video_object.cnn_visualiser_images_folder)

    logger.debug("Done")
    video_object.save()

if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/video_object.npy').item()
    list_of_global_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/preprocessing/global_fragments.npy').item()
    list_of_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/preprocessing/fragments.npy').item()
    list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
    params = NetworkParams(video.number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = None,
                                save_folder = video.accumulation_folder,
                                restore_folder = video.accumulation_folder,
                                image_size = video.identification_image_size,
                                video_path = video.video_path)
    net = ConvNetwork(params, training_flag = True)
    net.restore()
    first_global_fragment = list_of_global_fragments.global_fragments[0]

    for i in range(video.number_of_animals):
        for j in range(10):
            image = first_global_fragment.individual_fragments[i].images[j]
            image = np.expand_dims(image, 2)
            image = np.expand_dims(image, 0)
            label = first_global_fragment.individual_fragments[i].final_identity -1

            visualise(video, net, image, label)
