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
    h_fc_drop = buildFc(h_conv5_flat,w5,h5,d5,n_fc,1.0)
    h_relu = tf.nn.relu(h_fc_drop)
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

    # plt.figure
    # plt.subplot(121)
    # plt.imshow(np.reshape(X1_train[39999,:],[32,32]))
    # plt.subplot(122)
    # plt.imshow(np.reshape(X2_train[40000,:],[32,32]))
    # plt.show()
    # print Y_train
    resolution = np.prod(imsize)
    x1 = tf.placeholder(tf.float32, [None, resolution])
    x2 = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32)

    h1 = cnn_model(x1,imsize[1],imsize[2],imsize[0],keep_prob)
    tf.
    h2 = cnn_model(x2,imsize[1],imsize[2],imsize[0],keep_prob)


    # Define loss/eval/training functions
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    # lossFunc, Li, Lg, Q, Ew, diff = contrastive_loss(h1,h2,y)
    lossFunc= contrastive_loss2(h1,h2,y)
    contrastiveLoss = tf.reduce_mean(lossFunc)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    lr0 = 0.01
    gamma = 0.001
    power = 0.75
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(contrastiveLoss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(contrastiveLoss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(contrastiveLoss)

    # Monitor accuracy???

    ckpt_dir = args.ckpt_folder
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()
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

        indices = np.linspace(0, train_size*numIndiv, iter_per_epoch)
        indices = indices.astype('int')
        plt.ion()
        if args.train == 1:

            lossPlot = []
            lr = lr0
            for epoch_i in range(n_epochs):
                clusterPlot = {'h1':[],'h2':[]}
                lossEpoch = []
                for iter_i in range(iter_per_epoch - 1):
                    batch_x1s = X1_train[indices[iter_i]:indices[iter_i+1]]
                    batch_x2s = X2_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    # print batch_x1s[0,:]
                    feat1, feat2 = sess.run([h1,h2],
                                    feed_dict={
                                        x1: batch_x1s,
                                        x2: batch_x2s,
                                        keep_prob: 1.
                                    })
                    clusterPlot['h1'].append(feat1)
                    clusterPlot['h2'].append(feat2)

                    if iter_i % 250 == 0:
                        # print 'feat1 **********'
                        # print feat1[0:10,:]
                        # print 'feat2 **********'
                        # print feat2[0:10,:]
                        # print '****************'
                        loss = sess.run(contrastiveLoss,
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })

                        lossEpoch.append(loss)

                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.10f}".format(loss)


                    # sess.run(optimizer,
                    #         feed_dict={
                    #             x1: batch_x1s,
                    #             x2: batch_x2s,
                    #             y: batch_ys,
                    #             keep_prob: 1.,
                    #             learning_rate: lr
                    #         })

                lr = lr0 * (1 + gamma * epoch_i) ** -power
                print '-------------------'
                print lr
                print '-------------------'
                # print 'learning rate ****************'
                # print lr
                lossPlot.append(np.mean(lossEpoch))

                clusterPlot['h1'] = [inner for outer in clusterPlot['h1'] for inner in outer]
                clusterPlot['h2'] = [inner for outer in clusterPlot['h2'] for inner in outer]
                # print 'h1 **********'
                # print clusterPlot['h1']
                # print 'h2 **********'
                # print clusterPlot['h2']
                # print '****************'
                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)


                plotterSiamese(lossPlot,clusterPlot,Y1_train,Y2_train,numIndiv)

        if args.train == 0:

            print('Accuracy for (%s): ' % ckpt.model_checkpoint_path + str(sess.run(accuracy,
                                                             feed_dict={
                                                                 x: X_test,
                                                                 y: Y_test,
                                                                 keep_prob: 1.0
                                                             })))
        if args.train == 2:
            print "Implement weight visualisation"
