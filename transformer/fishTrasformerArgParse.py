#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

# path = 'matlabexports/imdb_5indiv_13000_9000_50_s1'
def get_data(path):
    # load data
    Fish = {}
    # read big matlab files
    # imdb_5indiv_13000_9000_50_s1
    # imdb_5indiv_15000_1000_s1
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'


    data = Fish['data'].astype(np.float32)
    # get image dimensions
    imsize = data[0].shape

    label = Fish['labels'].astype(np.int32)
    label = np.squeeze(label)
    # compute number of individuals
    numIndiv = len(list(set(label)))
    # get labels in {0,...,numIndiv-1}
    label = np.subtract(label,1)
    print label
    print 'splitting data in train, test and validation'
    N = 3000*numIndiv # of training data
    N_test = 500*numIndiv # of test data
    N_val = 500*numIndiv # validation data

    X_train = data[:N]
    X_test = data[N:N+N_test]
    X_valid = data[N+N_test:N+N_test+N_val]
    y_train = label[:N]
    y_test = label[N:N+N_test]
    y_valid = label[N+N_test:N+N_test+N_val]

    resolution = np.prod(imsize)

    X_train = np.reshape(X_train, [N, resolution])
    X_test = np.reshape(X_test, [N_test, resolution])
    X_valid = np.reshape(X_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    Y_train = dense_to_one_hot(y_train, n_classes=numIndiv)
    Y_valid = dense_to_one_hot(y_valid, n_classes=numIndiv)
    Y_test = dense_to_one_hot(y_test, n_classes=numIndiv)
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid, resolution, numIndiv

def build_model(resolution, numIndiv):
    # %% Graph representation of our network
    # %% Placeholders for imsize = height*width resolution and labels

    x = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None, numIndiv])

    # %% Since x is currently [batch, height*width], we need to reshape to a
    # 4-D tensor to use it in a convolutional graph.  If one component of
    # `shape` is the special value -1, the size of that dimension is
    # computed so that the total size remains constant.  Since we haven't
    # defined the batch dimension's shape yet, we use -1 to denote this
    # dimension should not change size.
    x_tensor = tf.reshape(x, [-1, imsize[1], imsize[2], imsize[0]])

    # %% We'll setup the two-layer localisation network to figure out the
    # %% parameters for an affine transformation of the input
    # %% Create variables for fully connected layer
    W_fc_loc1 = weight_variable([resolution, 20])
    b_fc_loc1 = bias_variable([20])
    # 6 is the number of parameters needed for the affine transformer
    W_fc_loc2 = weight_variable([20, 6])
    # Use identity transformation as starting point
    # the multiplication (2x2) matrix is set to be the identity and the
    # sum (translation) is [0,0]
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    # %% Define the two layer localisation network
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
    # %% We can add dropout for regularizing and to reduce overfitting like so:
    keep_prob = tf.placeholder(tf.float32)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    # %% Second layer
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

    # %% We'll create a spatial transformer module to identify discriminative
    # %% patches
    out_size = (imsize[1], imsize[2])
    h_trans = transformer(x_tensor, h_fc_loc2, out_size)

    # %% We'll setup the first convolutional layer
    filter_size = 3
    n_filters_1 = 16
    # Weight matrix is [height x width x input_channels x output_channels]
    W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

    # %% Bias is [output_channels]
    b_conv1 = bias_variable([n_filters_1])

    # %% Now we can build a graph which does the first layer of convolution:
    # we define our stride as batch x height x width x channels
    # instead of pooling, we use strides of 2 and more layers
    # with smaller filters.

    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(input=h_trans,
                     filter=W_conv1,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv1)

    # %% And just like the first layer, add additional layers to create
    # a deep net
    n_filters_2 = 64
    W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
    b_conv2 = bias_variable([n_filters_2])
    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(input=h_conv1,
                     filter=W_conv2,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv2)
    #
    #
    # # %% And why not a third...
    n_filters_3 = 64
    W_conv3 = weight_variable([filter_size, filter_size, n_filters_2, n_filters_3])
    b_conv3 = bias_variable([n_filters_3])
    h_conv3 = tf.nn.relu(
        tf.nn.conv2d(input=h_conv2,
                     filter=W_conv3,
                     strides=[1, 2, 2, 1],
                     padding='SAME') +
        b_conv3)

    # # %% And why not a fourth...
    # n_filters_4 = 16
    # W_conv4 = weight_variable([filter_size, filter_size, n_filters_3, n_filters_4])
    # b_conv4 = bias_variable([n_filters_4])
    # h_conv4 = tf.nn.relu(
    #     tf.nn.conv2d(input=h_conv3,
    #                  filter=W_conv4,
    #                  strides=[1, 2, 2, 1],
    #                  padding='SAME') +
    #     b_conv4)
    # %% We'll now reshape so we can connect to a fully-connected layer:
    # print resolution
    lastVolSize = 12*12*n_filters_3
    # lastVolSize = 7*7*n_filters_3
    h_conv4_flat = tf.reshape(h_conv3, [-1, lastVolSize])

    # %% Create a fully-connected layer:

    n_fc = 4096
    W_fc1 = weight_variable([lastVolSize, n_fc])
    b_fc1 = bias_variable([n_fc])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # %% second fc
    # n_fc2 = 256
    # W_fc2 = weight_variable([n_fc, n_fc2])
    # b_fc2 = bias_variable([n_fc2])
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # %% And finally our softmax layer:
    W_fc3 = weight_variable([n_fc, numIndiv])
    b_fc3 = bias_variable([numIndiv])
    y_logits = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

    # %% Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    opt = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    # opt = tf.train.FtrlOptimizer(0.01, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.5, use_locking=False, name='Ftrl')
    # opt = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.16, momentum=0.9, epsilon=1.0, use_locking=False, name='RMSProp')
    optimizer = opt.minimize(cross_entropy)
    grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

    # %% Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# # %% We now create a new session to actually perform the initialization the
# # variables:
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
#
# # %% We'll now train in minibatches and report accuracy, loss:
# iter_per_epoch = 100
# n_epochs = 1000
# train_size = N
#
# indices = np.linspace(0, train_size - 1, iter_per_epoch)
# indices = indices.astype('int')
#
# for epoch_i in range(n_epochs):
#     for iter_i in range(iter_per_epoch - 1):
#         batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
#         batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
#
#         if iter_i % 10 == 0:
#             loss = sess.run(cross_entropy,
#                             feed_dict={
#                                 x: batch_xs,
#                                 y: batch_ys,
#                                 keep_prob: 1.0
#                             })
#
#             print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))
#
#         sess.run(optimizer, feed_dict={
#             x: batch_xs, y: batch_ys, keep_prob: 0.8})
#
#     print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
#                                                      feed_dict={
#                                                          x: X_valid,
#                                                          y: Y_valid,
#                                                          keep_prob: 1.0
#                                                      })))
#     # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
#     # print(convolve[0].shape)
# # theta = sess.run(h_fc_loc2, feed_dict={
# #        x: batch_xs, keep_prob: 1.0})
# # print(theta[0])


if __name__ == '__main__':
    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='matlabexports/imdb_5indiv_13000_9000_50_s1.mat', type = str)
    parser.add_argument('--train', default=1, type=int)
    # parser.add_argument('--gpu', default=-1, type=int)
    # parser.add_argument('--epoch', default=100, type=int)
    # parser.add_argument('--batchsize', default=128, type=int)
    # parser.add_argument('--lr', default=0.01, type=float)
    # parser.add_argument('--gamma', default=0.001, type=float)
    # parser.add_argument('--power', default=0.75, type=float)
    # parser.add_argument('--momentum', default=0.9, type=float)
    #
    # parser.add_argument('--optimizer', default='SGD', type=str,
                        # choices=['SGD', 'Adam'])


    args = parser.parse_args()
    if args.train == 1 :
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = get_data(args.dataset)
