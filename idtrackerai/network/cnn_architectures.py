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
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import sys
from idtrackerai.utils.cnn_utils import *
from confapp import conf

''' original model'''
def cnn_model_0(images, classes, width, height, channels):
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
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    # return y_logits
    return y_logits, relu

''' model with 1 convolution '''
def cnn_model_1(images, classes, width, height, channels):
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
    # linearize weights for fully-connected layer
    resolutionS = w1 * h1
    conv1_flat = tf.reshape(relu1, [-1, resolutionS*n_filter1], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv1_flat, w1, h1, n_filter1, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' model with 2 convolutions '''
def cnn_model_2(images, classes, width, height, channels):
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
    # linearize weights for fully-connected layer
    resolutionS = w3 * h3
    conv3_flat = tf.reshape(relu2, [-1, resolutionS*n_filter3], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv3_flat, w3, h3, n_filter3, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' 4 conv'''
def cnn_model_3(images, classes, width, height, channels):
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
    # maxpool2d
    stride6 = [1,2,2,1]
    pool6 = 2
    pad6 = 'SAME'
    max_pool6, w6, h6 = maxpool2d('maxpool3',w5,h5, relu3, pool6,stride6,pad6)
    d6 = n_filter5
    # conv4
    filter_size7 = 5
    n_filter7 = 100
    stride7 = [1,1,1,1]
    pad7 = 'SAME'
    conv7, w7, h7, W7  = buildConv2D('conv4', w6, h6, d6, max_pool6, filter_size7, n_filter7, stride7, pad7)
    d7 = n_filter7
    # relu
    relu8 = reLU('relu4', conv7)
    # linearize weights for fully-connected layer
    resolutionS = w7 * h7
    conv7_flat = tf.reshape(relu8, [-1, resolutionS*d7], name = 'conv7_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv7_flat, w7, h7, d7, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' inverted model '''
def cnn_model_4(images, classes, width, height, channels):
    '''
    Gives predictions for a given set of images
    '''

    tf.summary.image('rawImages', images, max_outputs=10)
    # conv1
    filter_size1 = 5
    n_filter1 = 100
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
    n_filter5 = 16
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
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' no nonlinearlities '''
def cnn_model_5(images, classes, width, height, channels):
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
    conv5, w5, h5, W5  = buildConv2D('conv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    conv5_flat = tf.reshape(conv5, [-1, resolutionS*d5], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', fc_drop, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' smaller fully connected '''
def cnn_model_6(images, classes, width, height, channels):
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
    n_fc = 50
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' bigger fully connected '''
def cnn_model_7(images, classes, width, height, channels):
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
    n_fc = 200
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)
    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' two fully connected '''
def cnn_model_8(images, classes, width, height, channels):
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
    n_fc_1 = 100
    fc_drop_1, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc_1, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop_1)
    n_fc_2 = 100
    fc_drop_2, WFC_2 = buildSoftMax('fully-connected2', relu, n_fc_1, n_fc_2)
    relu = reLU('relu5', fc_drop_2)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc_2, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' two fully smaller connected '''
def cnn_model_9(images, classes, width, height, channels):
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
    n_fc_1 = 100
    fc_drop_1, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc_1, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop_1)
    n_fc_2 = 50
    fc_drop_2, WFC_2 = buildSoftMax('fully-connected2', relu, n_fc_1, n_fc_2)
    relu = reLU('relu5', fc_drop_2)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc_2, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' two fully bigger connected '''
def cnn_model_10(images, classes, width, height, channels):
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
    n_fc_1 = 100
    fc_drop_1, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc_1, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop_1)
    n_fc_2 = 200
    fc_drop_2, WFC_2 = buildSoftMax('fully-connected2', relu, n_fc_1, n_fc_2)
    relu = reLU('relu5', fc_drop_2)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc_2, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

''' even smaller fully connected '''
def cnn_model_11(images, classes, width, height, channels):
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
    n_fc = 10
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w5, h5, d5, n_fc, conf.KEEP_PROB)
    relu = reLU('relu4', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)

    return y_logits, relu
    # return y_logits, relu, (W1, W3, W5, WFC, WSoft)

def cnn_model_crossing_detector(images, classes, width, height, channels):
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
    # linearize weights for fully-connected layer
    resolutionS = w3 * h3
    conv5_flat = tf.reshape(relu2, [-1, resolutionS*n_filter3], name = 'conv5_reshape')
    # fully-connected 1
    n_fc = 100
    fc_drop, WFC = buildFc('fully-connected1', conv5_flat, w3, h3, n_filter3, n_fc, conf.KEEP_PROB)
    relu = reLU('relu1', fc_drop)
    y_logits, WSoft = buildSoftMax('fully_connected_pre_softmax', relu, n_fc, classes)
    return y_logits
