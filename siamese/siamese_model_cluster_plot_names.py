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
    h_conv1, w1, h1 = buildConv2D('Wconv1','Bconv1',width, height, 1, x_tensor, filter_size1, n_filter1, stride1, pad1)
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
    h_conv3, w3, h3 = buildConv2D('Wconv2','Bconv2',w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
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
    h_conv5, w5, h5 = buildConv2D('Wconv3','Bconv3',w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
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
    h_fc_drop = buildFc('W1','B1',h_conv5_flat,w5,h5,d5,n_fc,1.0)
    h_relu = tf.nn.relu(h_fc_drop)
    n_plotter = 2
    plotter = buildFc('W2','B2',h_relu, 100,1,1,n_plotter,1.0)

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
    parser.add_argument('--genuineRatio', default = 0.5, type = float)
    args = parser.parse_args()

    path = args.dataset
    num_train = args.num_train
    num_test = args.num_test
    num_valid = args.num_val
    G_ratio = args.genuineRatio

    # numIndiv, imsize, \
    # X1_train, X1_valid, X1_test,\
    # X2_train, X2_valid, X2_test,\
    # Y_train, Y_valid, Y_test,\
    # Y1_train, Y1_valid, Y1_test,\
    # Y2_train, Y2_valid, Y2_test,\
    # train_size = dataHelperPlot(path, num_train, num_test, num_valid)

    # numIndiv, imsize, \
    # X1_train, X1_valid, X1_test,\
    # X2_train, X2_valid, X2_test,\
    # Y_train, Y_valid, Y_test,\
    # Y1_train, Y1_valid, Y1_test,\
    # Y2_train, Y2_valid, Y2_test,\
    # train_size,\
    # totalNumG, totalNumI = dataHelperPlot2(path, num_train, num_test, num_valid)
    #
    # print "***********"
    # print("totalNumG " + str(totalNumG))
    # print("totalNumI " + str(totalNumI))


    numIndiv, imsize,\
    X1_train, X1_valid, X1_test,\
    X2_train, X2_valid, X2_test,\
    Y_train, Y_valid, Y_test,\
    Y1_train, Y1_valid, Y1_test,\
    Y2_train, Y2_valid, Y2_test,\
    train_size, totalNumG, totalNumI = dataHelperPlot3(path, num_train, num_test, num_valid, G_ratio)
    # numIndiv, imsize, \
    # X1_train, X1_valid, X1_test,\
    # X2_train, X2_valid, X2_test,\
    # Y_train, Y_valid, Y_test,\
    # Y1_train, Y1_valid, Y1_test,\
    # Y2_train, Y2_valid, Y2_test,\
    # train_size = genereateDataset(path, num_train, num_valid, num_test)

    # print "train ----------"
    # print np.shape(X1_train)
    # print np.shape(X2_train)
    # print np.shape(Y_train)
    # print Y1_train
    # print Y2_train
    # print np.unique(Y_train)
    #
    # print "valid ----------"
    # print np.shape(X1_valid)
    # print np.shape(X2_valid)
    # print np.shape(Y_valid)
    # print np.unique(Y_valid)
    #
    # print "test ----------"
    # print np.shape(X1_test)
    # print np.shape(X2_test)
    # print np.shape(Y_test)
    # print np.unique(Y_test)

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

    with tf.variable_scope('convolutions') as scope:
        h1 = cnn_model(x1,imsize[1],imsize[2],imsize[0],keep_prob)
        scope.reuse_variables()
        h2 = cnn_model(x2,imsize[1],imsize[2],imsize[0],keep_prob)


    # Define loss/eval/training functions
    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    # lossFunc, Li, Lg = contrastive_loss1(h1,h2,y)
    lossFunc= contrastive_loss1(h1,h2,y)
    contrastiveLoss = tf.div(tf.reduce_mean(lossFunc),2.)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    lr0 = 0.01
    gamma = 0.001
    power = 0.75
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(contrastiveLoss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(contrastiveLoss)
    optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(contrastiveLoss)

    # Monitor accuracy
    FPR, TPR, accuracy = computeROCAccuracy(h1, h2, y)

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
        # train_size = np.round(args.num_train/2)

        indices = np.linspace(0, train_size, iter_per_epoch)
        indices = indices.astype('int')
        plt.ion()
        if args.train == 1:

            if start == 0:
                ''' COMPUTE FEATURES, LOSS AND ACCURACY FOR TRAIN AND VAL BEFORE TRAINING '''
                print '*** Computing features loss and accuracy for the first time...'
                ''' training set '''
                print '-Training set-'

                clusterPlotTrain = {'h1':[],'h2':[]}
                lossEpoch = []
                accEpoch = []
                npFPRtrain = []
                npTPRtrain = []
                print ' Entering loop for batches'
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

                    if iter_i % 250 == 0:
                        loss, _, _, ROCAccTrain = sess.run([contrastiveLoss, FPR, TPR, accuracy],
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })

                        # loss, LI, LG, _, _, ROCAccTrain = sess.run([contrastiveLoss, Li, Lg, FPR, TPR, accuracy],
                        #                 feed_dict={
                        #                     x1: batch_x1s,
                        #                     x2: batch_x2s,
                        #                     y: batch_ys,
                        #                     keep_prob: 1.0
                        #                 })
                        # print batch_x1s
                        # print batch_x2s
                        # print feat1
                        # print feat2
                        # print loss
                        # print LI
                        # print LG

                        # FPR, TPR = accuracy(feat1,feat2,batch_ys)
                        # npFPR = sess.run(FPR)
                        # npTPR = sess.run(TPR)
                        # acc = np.trapz(npTPR,npFPR)

                        print "  Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.10f}".format(loss) + ", Minibatch Acc= " + "{:.10f}".format(ROCAccTrain)
                        lossEpoch.append(loss)
                        accEpoch.append(ROCAccTrain)

                    clusterPlotTrain['h1'].append(feat1)
                    clusterPlotTrain['h2'].append(feat2)
                    # npFPRtrain.append(npFPR)
                    # npTPRtrain.append(npTPR)

                print' Concatenating minibatches... '
                clusterPlotTrain['h1'] = [inner for outer in clusterPlotTrain['h1'] for inner in outer]
                clusterPlotTrain['h2'] = [inner for outer in clusterPlotTrain['h2'] for inner in outer]
                lossTrainPlot = [np.mean(lossEpoch)]
                accTrainPlot = [np.mean(accEpoch)]
                # npFPRtrain = [inner for outer in npFPRtrain for inner in outer]
                # npTPRtrain = [inner for outer in npTPRtrain for inner in outer]

                ''' validation set '''
                print '-Validation set-'
                feat1val, feat2val = sess.run([h1,h2],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    keep_prob: 1.
                                })
                clusterPlotVal = {'h1':feat1val,'h2':feat2val}

                lossVal, FPRVal, TPRVal, ROCAccVal = sess.run([contrastiveLoss, FPR, TPR, accuracy],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    y: Y_valid,
                                    keep_prob: 1.0
                                })
                lossValPlot = [lossVal]
                accValPlot = [ROCAccVal]
                # FPRval, TPRval = accuracy(feat1val,feat2val,Y_valid)
                # npFPRval = sess.run(FPRval)
                # npTPRval = sess.run(TPRval)
                # accVal = np.trapz(npTPRval,npFPRval)

                print "  No training yet , Validation Loss= " + "{:.10f}".format(lossVal) + ", Validation Acc= " + "{:.10f}".format(ROCAccVal)

                ''' plotting '''
                plotterSiameseAccuracy(
                    lossTrainPlot, lossValPlot,
                    accTrainPlot, accValPlot,
                    TPRVal, FPRVal,
                    clusterPlotTrain, clusterPlotVal,
                    Y1_train, Y1_valid,
                    Y2_train, Y2_valid,
                    numIndiv)
                print 'Saving figure...'
                figname = ckpt_dir + '/result_start.pdf'
                plt.savefig(figname)

            ''' TRAINING ****************************************************'''
            print('*** Starting training ')
            if start != 0:
                lossTrainPlot = []
                accTrainPlot = []
                lossValPlot = []
                accValPlot = []

            lr = lr0
            for epoch_i in range(n_epochs):
                clusterPlotTrain = {'h1':[],'h2':[]}
                lossEpoch = []
                accEpoch = []
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
                    clusterPlotTrain['h1'].append(feat1)
                    clusterPlotTrain['h2'].append(feat2)

                    if iter_i % 250 == 0:
                        # print 'feat1 **********'
                        # print feat1[0:10,:]
                        # print 'feat2 **********'
                        # print feat2[0:10,:]
                        # print '****************'
                        loss, _, _, ROCAccTrain = sess.run([contrastiveLoss, FPR, TPR, accuracy],
                                        feed_dict={
                                            x1: batch_x1s,
                                            x2: batch_x2s,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })

                        # FPR, TPR = accuracy(feat1,feat2,batch_ys)
                        # npFPR = sess.run(FPR)
                        # npTPR = sess.run(TPR)
                        # acc = np.trapz(npTPR,npFPR)

                        print "  Epoch " + str(epoch_i) + ", iter " + str(iter_i) + ", Minibatch Loss= " + "{:.10f}".format(loss) + ", Minibatch Acc= " + "{:.10f}".format(ROCAccTrain)
                        lossEpoch.append(loss)
                        accEpoch.append(ROCAccTrain)

                    lr = lr0 * (1 + gamma * epoch_i) ** -power
                    # optimizer.lr = args.lr * (1 + args.gamma * epoch) ** -args.power
                    sess.run(optimizer,
                            feed_dict={
                                x1: batch_x1s,
                                x2: batch_x2s,
                                y: batch_ys,
                                keep_prob: 1.,
                                learning_rate: lr
                            })


                print 'learning rate ****************'
                print lr

                lossTrainPlot.append(np.mean(lossEpoch))
                accTrainPlot.append(np.mean(accEpoch))

                clusterPlotTrain['h1'] = [inner for outer in clusterPlotTrain['h1'] for inner in outer]
                clusterPlotTrain['h2'] = [inner for outer in clusterPlotTrain['h2'] for inner in outer]
                # print 'h1 **********'
                # print clusterPlotTrain['h1']
                # print 'h2 **********'
                # print clusterPlotTrain['h2']
                # print '****************'

                ''' Compute loss, accuracy and features for the validations for this epoch '''
                print '   Computing loss, accuracy and feautres for the validation set...'
                feat1val, feat2val = sess.run([h1,h2],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    keep_prob: 1.
                                })
                clusterPlotVal = {'h1':feat1val,'h2':feat2val}

                lossVal, FPRVal, TPRVal, ROCAccVal = sess.run([contrastiveLoss, FPR, TPR, accuracy],
                                feed_dict={
                                    x1: X1_valid,
                                    x2: X2_valid,
                                    y: Y_valid,
                                    keep_prob: 1.0
                                })
                lossValPlot.append(lossVal)
                accValPlot.append(ROCAccVal)
                print "  Epoch " + str(epoch_i) + ", Validation Loss= " + "{:.10f}".format(lossVal) + ", Validation Acc= " + "{:.10f}".format(ROCAccVal)

                ''' plotting '''
                plotterSiameseAccuracy(
                    lossTrainPlot, lossValPlot,
                    accTrainPlot, accValPlot,
                    TPRVal, FPRVal,
                    clusterPlotTrain, clusterPlotVal,
                    Y1_train, Y1_valid,
                    Y2_train, Y2_valid,
                    numIndiv)
                # plotterSiameseAccuracy(lossTrainPlot,clusterPlot,Y1_train,Y2_train,numIndiv,falsePositiveRate,truePositiveRate)
                print 'Saving model...'
                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)

                print 'Saving figure...'
                figname = ckpt_dir + '/result_' + str(epoch_i) + '.pdf'
                plt.savefig(figname)
                print '-------------------------------'



        if args.train == 0:

            featTest = sess.run(h1,
                            feed_dict={
                                x1: X1_valid,
                                keep_prob: 1.
                            })

            plotterSiameseTest(featTest,Y1_valid,numIndiv)
            plt.draw()

        if args.train == 2:
            print "Implement weight visualisation"
