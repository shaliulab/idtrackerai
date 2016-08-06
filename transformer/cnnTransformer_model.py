import os
import sys
sys.path.append('../utils')
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_utils import *
from input_data import *
from cnn_utils import *
from spTrans_utils import *
import argparse
import h5py
import numpy as np
import time


def model(x,y,width, height, channels, classes, keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels])
    # sp1
    n_loc = 50
    h_trans, w0, h0 = spTrans(x,x_tensor,width, height, channels, n_loc,keep_prob)
    # conv1
    filter_size1 = 5
    n_filter1 = 15
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    h_conv1, w1, h1 = buildConv2D(w0, h0, 1, h_trans, filter_size1, n_filter1, stride1, pad1)
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
    y_logits = buildSoftMax(h_relu,n_fc,classes)

    return y_logits, h_trans, h_conv5

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

    numIndiv, imsize, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = dataHelper(path, num_train, num_test, num_valid)
    resolution = np.prod(imsize)
    classes = numIndiv
    x = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)

    y_logits, h_trans, h_conv5 = model(x,y,imsize[1],imsize[2],imsize[0],classes,keep_prob)


    # Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    # opt = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    opt = tf.train.GradientDescentOptimizer(0.005)
    optimizer = opt.minimize(cross_entropy)

    # Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

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
        train_size = args.num_train

        indices = np.linspace(0, train_size*numIndiv - 1, iter_per_epoch)
        indices = indices.astype('int')

        plt.ion()

        if args.train == 1:
            # print "Start from:", start
            for epoch_i in range(n_epochs):
                for iter_i in range(iter_per_epoch - 1):
                    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    if iter_i % 250 == 0:
                        loss, acc = sess.run([cross_entropy,accuracy],
                                        feed_dict={
                                            x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                        # convolve = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                        # print(convolve[0])
                    sess.run(optimizer, feed_dict={
                        x: batch_xs, y: batch_ys, keep_prob: 1})
                # convolve = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0])

                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)

                print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                                                                 feed_dict={
                                                                     x: X_valid,
                                                                     y: Y_valid,
                                                                     keep_prob: 1.0
                                                                 })))

                testInd = 0
                theta = sess.run(h_trans, feed_dict={
                       x: X_valid, keep_prob: 1.0})
                # print(theta[0])
                testImage = np.reshape(X_valid[testInd], [imsize[1], imsize[2]])
                testImage2 = np.reshape(X_valid[testInd+1], [imsize[1], imsize[2]])
                testImage3 = np.reshape(X_valid[testInd+2], [imsize[1], imsize[2]])

                plt.close()
                plt.figure
                plt.subplot(321)
                plt.imshow(testImage)

                plt.subplot(322)
                plt.imshow(np.squeeze(theta[testInd]), interpolation='none')


                plt.figure
                plt.subplot(323)
                plt.imshow(testImage2)

                plt.subplot(324)
                plt.imshow(np.squeeze(theta[testInd+1]),interpolation='none')


                plt.figure
                plt.subplot(325)
                plt.imshow(testImage3)

                plt.subplot(326)
                plt.imshow(np.squeeze(theta[testInd+2]),interpolation='none')

                plt.draw()
                plt.pause(1)
                # time.sleep(1)
                # plt.ioff()
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
            plt.ioff()
            testInd = 1500
            theta = sess.run(h_trans, feed_dict={
                   x: X_test, keep_prob: 1.0})
            # # print(theta[0])
            # testImage = np.reshape(X_test[testInd], [imsize[1], imsize[2]])
            # plt.subplot(121)
            # plt.imshow(testImage)
            # # transf = tf.reshape(theta[0], [2,3])
            # # print theta[0].shape
            # plt.subplot(122)
            # plt.imshow(np.squeeze(theta[testInd]))


            testImage = np.reshape(X_test[testInd], [imsize[1], imsize[2]])
            testImage2 = np.reshape(X_test[testInd+1], [imsize[1], imsize[2]])
            testImage3 = np.reshape(X_test[testInd+2], [imsize[1], imsize[2]])

            plt.close()
            plt.figure
            plt.subplot(321)
            plt.imshow(testImage, cmap = 'gray')

            plt.subplot(322)
            plt.imshow(np.squeeze(theta[testInd]), interpolation='none',cmap = 'gray')


            plt.figure
            plt.subplot(323)
            plt.imshow(testImage2,cmap = 'gray')

            plt.subplot(324)
            plt.imshow(np.squeeze(theta[testInd+1]),interpolation='none',cmap = 'gray')


            plt.figure
            plt.subplot(325)
            plt.imshow(testImage3,cmap = 'gray')

            plt.subplot(326)
            plt.imshow(np.squeeze(theta[testInd+2]),interpolation='none',cmap = 'gray')
            plt.show()
