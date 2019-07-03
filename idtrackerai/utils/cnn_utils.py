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

import os
import sys
from idtrackerai.utils.py_utils import  *
import tensorflow as tf
import numpy as np
import re

if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.cnn_utils")

# ****************************************************************************
# Tensorboard
# *****************************************************************************

# def _activation_summary(x):
#   """Taken from [1]_
#   Helper to create summaries for activations.
#   Creates a summary that provides a histogram of activations.
#   Creates a summary that measure the sparsity of activations.
#   .. [1] https://www.programcreek.com/python/example/90327/tensorflow.scalar_summary
#   """
#   # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#   # session. This helps the clarity of presentation on tensorboard.
#   tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
#   tf.summary.histogram(tensor_name + '/activations', x)
#   tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
#
# def put_kernels_on_grid(kernel, grid, pad=1):
#     '''Taken from [2]_
#     Visualize conv. features as an image (mostly for the 1st layer).
#     Place kernel into a grid, with some paddings between adjacent filters.
#     Args:
#       kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
#       (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
#                            User is responsible of how to break into two multiples.
#       pad:               number of black pixels around each filter (between them)
#
#     Return:
#       Tensor of shape [(Y+pad)*grid_Y, (X+pad)*grid_X, NumChannels, 1].
#
#     .. [2] https://gist.github.com/kukuruza/03731dc494603ceab0c5
#     '''
#     grid_Y, grid_X = grid
#     # pad X and Y
#     x1 = tf.pad(kernel, tf.constant( [[pad,pad],[pad,pad],[0,0],[0,0]] ))
#     # X and Y dimensions, w.r.t. padding
#     Y = kernel.get_shape()[0] + 2*pad
#     X = kernel.get_shape()[1] + 2*pad
#     NumChannels = kernel.get_shape()[2]
#     # put NumKernels to the 1st dimension
#     x2 = tf.transpose(x1, (3, 0, 1, 2))
#     # organize grid on Y axis
#     x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, NumChannels]))
#     # switch X and Y axes
#     x4 = tf.transpose(x3, (0, 2, 1, 3))
#     # organize grid on X axis
#     x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, NumChannels]))
#     # back to normal order (not combining with the next step for clarity)
#     x6 = tf.transpose(x5, (2, 1, 3, 0))
#     # to tf.summary.image order [batch_size, height, width, channels],
#     #   where in this case batch_size == 1
#     x7 = tf.transpose(x6, (3, 0, 1, 2))
#     # scale to [0, 1]
#     x_min = tf.reduce_min(x7)
#     x_max = tf.reduce_max(x7)
#     x8 = (x7 - x_min) / (x_max - x_min)
#
#     return x8

# ****************************************************************************
# CNN wrappers
# *****************************************************************************

def computeVolume(width, height, strides):
    c1 = float(strides[1])
    c2 = float(strides[2])
    widthS = int(np.ceil(width/c1))
    heightS = int(np.ceil(height/c2))
    return widthS, heightS

def buildConv2D(scopeName, inputWidth, inputHeight, inputDepth, inputConv ,filter_size, n_filters, stride, pad):
    w,h = computeVolume(inputWidth, inputHeight, stride)
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [filter_size, filter_size, inputDepth, n_filters],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0.)
            )
        b = tf.get_variable(
            'biases',
            [n_filters],
            # initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1)
            initializer=tf.constant_initializer(0.0)
            )
        conv = tf.nn.conv2d(
                    input=inputConv,
                     filter=W,
                     strides=stride,
                     padding=pad)
        convb = tf.nn.bias_add(conv, b, name = scope.name)

        # _activation_summary(convb)
        # 
        # with tf.variable_scope('visualization') :
        #     grid_x = int(np.sqrt(n_filters))
        #     grid_y = grid_x  # to get a square grid for 64 conv1 features
        #     WtoPlot = tf.slice(W, [0, 0, 0, 0], [filter_size, filter_size, 1, n_filters])
        #     grid = put_kernels_on_grid(WtoPlot, (grid_y, grid_x))
        #     tf.summary.image(scopeName + '/features', grid, max_outputs=1)
        #     convbToPlot = tf.slice(convb, [0, 0, 0, 0], [-1, w, h, 1])
        #     # this will display random images
        #     tf.summary.image(scopeName + '/output', convbToPlot, max_outputs=10)
    return convb, w, h, W

def maxpool2d(name,inputWidth, inputHeight, inputPool, pool=2 ,
                stride=[1,2,2,1] ,pad='VALID'):
    max_pool = tf.nn.max_pool(inputPool,
        ksize=[1, pool, pool, 1],
        strides=stride,
        padding=pad,
        name = name
        )
    w, h = computeVolume(inputWidth, inputHeight, stride)
    return max_pool, w, h

def buildFc(scopeName, inputFc, height, width, n_filters, n_fc, keep_prob):
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [height * width * n_filters, n_fc],
            initializer=tf.contrib.layers.xavier_initializer(seed=0.)
            )
        b = tf.get_variable(
            'biases',
            [n_fc],
            initializer=tf.constant_initializer(0.0)
            )
        fc = tf.add(tf.matmul(inputFc, W), b)
        fc_drop = tf.nn.dropout(fc, keep_prob, name = scope.name)
        # _activation_summary(fc_drop)
    return fc_drop, W

def reLU(scopeName, inputRelu):
    with tf.variable_scope(scopeName) as scope:
        relu = tf.nn.relu(inputRelu, name = scope.name)
        # _activation_summary(relu)
    return relu

def buildSoftMax(scopeName, inputSoftMax, n_fc, classes):
    with tf.variable_scope(scopeName) as scope:
        W = tf.get_variable(
            'weights',
            [n_fc, classes],
            initializer=tf.contrib.layers.xavier_initializer(seed=0.)
            )
        b = tf.get_variable(
            'biases',
            [classes],
            initializer=tf.constant_initializer(0.0)
            )
        logits = tf.add(tf.matmul(inputSoftMax, W), b, name = scope.name)
        # _activation_summary(logits)
    return logits, W
