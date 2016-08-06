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

    return h_relu

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

    numIndiv, imsize, X1_train, X1_valid, X1_test, X2_train, X2_valid, X2_test, Y_train, Y_valid, Y_test = dataHelper(path, num_train, num_test, num_valid)
    resolution = np.prod(imsize)
    classes = numIndiv
    x1 = tf.placeholder(tf.float32, [None, resolution])
    x2 = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    h1 = cnn_model(x1,imsize[1],imsize[2],imsize[0],keep_prob)
    h2 = cnn_model(x2,imsize[1],imsize[2],imsize[0],keep_prob)


    # Define loss/eval/training functions
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    lossFunc, Li, Lg, Q, Ew, diff = contrastive_loss(h1,h2,y)
    contrastiveLoss = tf.reduce_mean(lossFunc)
    # opt = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    opt = tf.train.GradientDescentOptimizer(0.00001)
    optimizer = opt.minimize(contrastiveLoss)

    # Monitor accuracy
    # correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ckpt_dir = args.ckpt_folder
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()
    train = 1
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

        start = global_step.eval() # get last global_step
        print "Start from:", start

        # %% We'll now train in minibatches and report accuracy, loss:
        iter_per_epoch = args.itsPerEpoch
        n_epochs = args.num_epochs
        train_size = args.num_train/2
        print train_size

        indices = np.linspace(0, train_size*numIndiv - 1, iter_per_epoch)
        indices = indices.astype('int')
        plt.ion()
        if args.train == 1:
            # print "Start from:", start
            lossPlot = []
            for epoch_i in range(n_epochs):
                lossEpoch = []
                for iter_i in range(iter_per_epoch - 1):
                    batch_x1s = X1_train[indices[iter_i]:indices[iter_i+1]]
                    batch_x2s = X2_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]
                    # print batch_ys


                    if iter_i % 250 == 0:
                        loss = sess.run(contrastiveLoss,
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        lossEpoch.append(loss)
                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.6f}".format(loss)
                        # Lprints = sess.run([Li,Lg], feed_dict={x1: batch_x1s, x2: batch_x2s, y: batch_ys, keep_prob: 1.})
                        # print(Lprints[0],Lprints[1])

                    sess.run(optimizer, feed_dict={
                        x1: batch_x1s, x2: batch_x2s, y: batch_ys, keep_prob: 1.})

                lossPlot.append(np.mean(lossEpoch))
                # convolve = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0])

                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)

                plt.close()
                plt.figure
                plt.plot(lossPlot)

                plt.draw()
                plt.pause(1)

                # print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                #                                                  feed_dict={
                #                                                      x: X_valid,
                #                                                      y: Y_valid,
                #                                                      keep_prob: 1.0
                #                                                  })))
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)
        if args.train == 0:

            print('Accuracy for (%s): ' % ckpt.model_checkpoint_path + str(sess.run(accuracy,
                                                             feed_dict={
                                                                 x: X_test,
                                                                 y: Y_test,
                                                                 keep_prob: 1.0
                                                             })))
        if args.train == 2:
            print "Implement weight visualisation"
