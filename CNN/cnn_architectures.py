from __future__ import division
import sys
sys.path.append('../utils')
import tensorflow as tf
from tf_utils import *
from cnn_utils import *
import numpy as np


def inference(images, width, height, channels, classes, keep_prob):
    '''
    Gives predictions for a given set of images
    '''
    with tf.variable_scope('Input') as scope:
        images_tensor = tf.reshape(images, [-1, width, height, channels],name='images')
        with tf.variable_scope('visualization'):
            # scale weights to [0 1], type is still float
            x_min = tf.reduce_min(images_tensor)
            x_max = tf.reduce_max(images_tensor)
            # images_tensor_0_to_1 = (images_tensor - x_min) / (x_max - x_min)
            images_tensor_0_to_1 = images_tensor / 255

            # to tf.summary.image format [batch_size, height, width, channels]
            # images_placeholder_transposed = tf.transpose (images_placeholder_0_to_1, [0, 2, 3, 1])

            # this will display random images
            tf.summary.image('_rawImages', images_tensor_0_to_1, max_outputs=10)
    # conv1
    filter_size1 = 5
    n_filter1 = 16
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    conv1, w1, h1, W1 = buildConv2D('conv1', width, height, 1, images_tensor_0_to_1, filter_size1, n_filter1, stride1, pad1)
    # maxpool2d
    stride2 = [1,2,2,1]
    pool2 = 2
    pad2 = 'SAME'
    max_pool2, w2, h2 = maxpool2d('maxpool1',w1,h1, conv1, pool2,stride2,pad2)
    d2 = n_filter1
    # conv2
    filter_size3 = 5
    n_filter3 = 64
    stride3 = [1,1,1,1]
    pad3 = 'SAME'
    conv3, w3, h3, W3 = buildConv2D('conv2', w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
    # maxpool2d
    stride4 = [1,2,2,1]
    pool4 = 2
    pad4 = 'SAME'
    max_pool4, w4, h4 = maxpool2d('maxpool2',w3,h3, conv3, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    conv5, w5, h5, W5 = buildConv2D('conv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    conv5_flat = tf.reshape(conv5, [-1, resolutionS*d5], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, keep_prob)
    relu = reLU('relu1', fc_drop)
    y_logits = buildSoftMax('softmax1', relu, n_fc, classes)

    return y_logits, relu, (W1,W3,W5)

def inference1(images, width, height, channels, classes, keep_prob):
    '''
    Gives predictions for a given set of images
    '''
    

    with tf.variable_scope('Input') as scope:
        images_tensor = tf.reshape(images, [-1, width, height, channels],name='images')
        with tf.variable_scope('visualization'):
            # scale weights to [0 1], type is still float
            # x_min = tf.reduce_min(images_tensor)
            # x_max = tf.reduce_max(images_tensor)
            # images_tensor_0_to_1 = (images_tensor - x_min) / (x_max - x_min)
            # images_tensor_0_to_1 = images_tensor / 255

            # to tf.summary.image format [batch_size, height, width, channels]
            # images_placeholder_transposed = tf.transpose (images_placeholder_0_to_1, [0, 2, 3, 1])

            # this will display random images
            tf.summary.image('_rawImages', images_tensor, max_outputs=10)
    # conv1
    filter_size1 = 5
    n_filter1 = 16
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    conv1, w1, h1, W1 = buildConv2D('conv1', width, height, 1, images_tensor, filter_size1, n_filter1, stride1, pad1)
    # relu
    relu1 = reLU('relu1', conv1)
    # maxpool2d
    stride2 = [1,2,2,1]
    pool2 = 2
    pad2 = 'SAME'
    max_pool2, w2, h2 = maxpool2d('maxpool1',w1,h1, relu1, pool2,stride2,pad2)
    d2 = n_filter1
    # conv2
    filter_size3 = 5
    n_filter3 = 64
    stride3 = [1,1,1,1]
    pad3 = 'SAME'
    conv3, w3, h3, W3 = buildConv2D('conv2', w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
    # relu
    relu2 = reLU('relu2', conv3)
    # maxpool2d
    stride4 = [1,2,2,1]
    pool4 = 2
    pad4 = 'SAME'
    max_pool4, w4, h4 = maxpool2d('maxpool2',w3,h3, relu2, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    conv5, w5, h5, W5 = buildConv2D('conv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # relu
    relu3 = reLU('relu3', conv5)
    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    conv5_flat = tf.reshape(relu3, [-1, resolutionS*d5], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, keep_prob)
    relu = reLU('relu1', fc_drop)
    y_logits = buildSoftMax('softmax1', relu, n_fc, classes)

    return y_logits, relu, (W1,W3,W5)
