import tensorflow as tf
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from spatial_transformer import transformer

# Spatian transformer layer
def spTrans(x,x_tensor,width, height, channels, n_loc,keep_prob):
    resolution = width * height * channels
    W_fc_loc1 = weight_variable([resolution, n_loc])
    b_fc_loc1 = bias_variable([n_loc])

    W_fc_loc2 = weight_variable([n_loc, 6])
    # Use identity transformation as starting point
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    # Two layer localisation network
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
    # dropout (reduce overfittin)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    # %% Second layer
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
    # spatial transformer
    outWidth = width
    outHeight = height
    out_size = (outWidth, outHeight)
    h_trans = transformer(x_tensor, h_fc_loc2, out_size)

    return h_trans, outWidth, outHeight
