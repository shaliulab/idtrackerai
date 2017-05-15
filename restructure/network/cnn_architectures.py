from __future__ import division
import sys
sys.path.append('./utils')
import tensorflow as tf
from tf_utils import *
from cnn_utils import *
import numpy as np

IMAGE_SIZE = (32,32,1)
width = IMAGE_SIZE[0]
height = IMAGE_SIZE[1]
channels = IMAGE_SIZE[2]
keep_prob = 1.0

def cnn_model_2(images, classes):
    '''
    Gives predictions for a given set of images
    '''

    tf.summary.image('rawImages', images, max_outputs=10)
    # conv1
    filter_size = 5
    n_filter = 16
    stride = [1,1,1,1]
    pad = 'SAME'
    conv1, w, h = buildConv2D('conv1', width, height, 1, images, filter_size, n_filter, stride, pad)
    # relu
    relu1 = reLU('relu1', conv1)
    # maxpool2d
    stride = [1,2,2,1]
    pool = 2
    pad = 'SAME'
    max_pool1, w, h = maxpool2d('maxpool1',w,h, relu1, pool,stride,pad)
    d = n_filter
    # conv2
    filter_size = 5
    n_filter = 64
    stride = [1,1,1,1]
    pad = 'SAME'
    conv2, w, h = buildConv2D('conv2', w, h, d, max_pool1, filter_size, n_filter, stride, pad)
    # relu
    relu2 = reLU('relu2', conv2)
    # maxpool2d
    stride = [1,2,2,1]
    pool = 2
    pad = 'SAME'
    max_pool2, w, h = maxpool2d('maxpool2',w,h, relu2, pool,stride,pad)
    d = n_filter
    # conv3
    filter_size = 5
    n_filter = 100
    stride = [1,1,1,1]
    pad = 'SAME'
    conv3, w, h = buildConv2D('conv3', w, h, d, max_pool2, filter_size, n_filter, stride, pad)
    d = n_filter
    # relu
    relu3 = reLU('relu3', conv3)
    # maxpool2d
    stride = [1,2,2,1]
    pool = 2
    pad = 'SAME'
    max_pool3, w, h = maxpool2d('maxpool3',w,h, relu3, pool,stride,pad)
    d = n_filter
    # conv4
    filter_size = 5
    n_filter = 100
    stride = [1,1,1,1]
    pad = 'SAME'
    conv4, w, h = buildConv2D('conv4', w, h, d, max_pool3, filter_size, n_filter, stride, pad)
    d = n_filter
    # relu
    relu4 = reLU('relu4', conv4)
    # maxpool2d
    stride = [1,2,2,1]
    pool = 2
    pad = 'SAME'
    max_pool4, w, h = maxpool2d('maxpool4',w,h, relu4, pool,stride,pad)
    d = n_filter
    # conv4
    filter_size = 5
    n_filter = 50
    stride = [1,1,1,1]
    pad = 'SAME'
    conv5, w, h = buildConv2D('conv5', w, h, d, max_pool4, filter_size, n_filter, stride, pad)
    d = n_filter
    # relu
    relu5 = reLU('relu5', conv5)
    # linearize weights for fully-connected layer
    # resolutionS = w * h
    # conv5_flat = tf.reshape(relu3, [-1, resolutionS*d5], name = 'conv5_reshape')
    conv5_flat = tf.flatten(conv5)
    # fully-connected 1
    n_fc = 100
    fc_drop = buildFc('fully-connected1', conv5_flat, w, h, d, n_fc, keep_prob)
    relu = reLU('relu6', fc_drop)
    y_logits = buildSoftMax('softmax1', relu, n_fc, classes)

    return y_logits, conv5_flat

def cnn_model(images, classes):
    '''
    Gives predictions for a given set of images
    '''

    tf.summary.image('rawImages', images, max_outputs=10)
    # conv1
    filter_size1 = 5
    n_filter1 = 16
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    conv1, w1, h1, W1 = buildConv2D('conv1', width, height, 1, images, filter_size1, n_filter1, stride1, pad1)
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
    conv5, w5, h5, W5  = buildConv2D('conv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # relu
    relu3 = reLU('relu3', conv5)
    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    conv5_flat = tf.reshape(relu3, [-1, resolutionS*d5], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, keep_prob)
    relu = reLU('relu1', fc_drop)
    y_logits, WSoft = buildSoftMax('softmax1', relu, n_fc, classes)

    return y_logits, relu, (W1, W3, W5, WFC, WSoft)
