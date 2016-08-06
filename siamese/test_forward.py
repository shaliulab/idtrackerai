import os
import sys
sys.path.append('../utils')
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_utils import *
from input_data_siamese import *
from cnn_utils import *
from siamese_utils import *
import argparse
import h5py
import numpy as np
def cnn_model(x,width, height, channels, keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels])
    # conv1
    filter_size1 = 5
    n_filter1 = 15
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    h_conv1, w1, h1 = buildConv2D(width, height, 1, x_tensor, filter_size1, n_filter1, stride1, pad1)
    # maxpool2d
    stride2 = [1,2,2,1]
    pool2 = 2
    pad2 = 'SAME'
    max_pool2, w2, h2 = maxpool2d(w1,h1, h_conv1, pool2,stride2,pad2)
    d2 = n_filter1
    # conv2
    filter_size3 = 5
    n_filter3 = 50
    stride3 = [1,1,1,1]
    pad3 = 'SAME'
    h_conv3, w3, h3 = buildConv2D(w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
    # maxpool2d
    stride4 = [1,2,2,1]
    pool4 = 2
    pad4 = 'SAME'
    max_pool4, w4, h4 = maxpool2d(w3,h3, h_conv3, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    h_conv5, w5, h5 = buildConv2D(w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # # maxpool2d
    # stride6 = [1,2,2,1]
    # pool6 = 2
    # max_pool6, w6, h6 = maxpool2d(w5,h5, h_conv5, pool6,stride6,'VALID')
    # d6 = n_filter5

    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    h_conv5_flat = tf.reshape(h_conv5, [-1, resolutionS*d5])
    # fully-connected 1
    n_fc = 100
    h_fc_drop = buildFc(h_conv5_flat,w5,h5,d5,n_fc,keep_prob)
    h_relu = tf.nn.relu(h_fc_drop)
    # y_logits = buildSoftMax(h_relu,n_fc,classes)

    n_plotter = 2
    plotter = buildFc(h_relu, 100,1,1,n_plotter,1.0)

    return plotter

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='matlabexports/imdb_5indiv_3000_1000_s1.mat', type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--num_train', default = 2500, type = int)
    parser.add_argument('--num_val', default = 500, type = int)
    parser.add_argument('--num_test', default = 1000, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--itsPerEpoch', default = 100, type = int)
    args = parser.parse_args()

    path = args.dataset
    num_train = args.num_train
    num_test = args.num_test
    num_valid = args.num_val

    numIndiv, imsize, \
    X1_train, X1_valid, X1_test,\
    X2_train, X2_valid, X2_test,\
    Y_train, Y_valid, Y_test,\
    Y1_train, Y1_valid, Y1_test,\
    Y2_train, Y2_valid, Y2_test = dataHelperPlot(path, num_train, num_test, num_valid)

    resolution = np.prod(imsize)
    x1 = tf.placeholder(tf.float32, [None, resolution])
    x2 = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32)

    h1 = cnn_model(x1,imsize[1],imsize[2],imsize[0],keep_prob)
    h2 = cnn_model(x2,imsize[1],imsize[2],imsize[0],keep_prob)

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()
        batch_x1s = X1_train[0:2]
        batch_x2s = X2_train[0:2]
        feat1, feat2 = sess.run([h1,h2],
                        feed_dict={
                            x1: batch_x1s,
                            x2: batch_x2s,
                            keep_prob: 1.
                        })

        print 'feat1 **********'
        print feat1
        print 'feat2 **********'
        print feat2
        print '****************'
