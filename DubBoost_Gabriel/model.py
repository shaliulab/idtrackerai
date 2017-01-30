import os
import sys
sys.path.append('../utils')
import tensorflow as tf
from tf_utils import *
from input_data import *
from loadData import *
from cnn_utils import *
import argparse
import h5py
import numpy as np


def modelNoCorr(x, classes, keep_prob):
    h_relu = x
    n_fc = 40
    # fully-connected 1
    # n_fc = 2000
    # h_fc_drop = buildFc('W1','b1',x,40,1,1,n_fc,keep_prob)
    # h_relu = tf.nn.relu(h_fc_drop)
    y_logits, W_docs, b_docs = buildSoftMaxWeights(h_relu,n_fc,classes)

    return y_logits, W_docs, b_docs

def modelCorr(x, classes, keep_prob):
    # fully-connected 1
    n_fc = 100000
    h_fc_drop = buildFc('W1','b1',x,40,1,1,n_fc,keep_prob)
    h_relu = tf.nn.relu6(h_fc_drop)
    y_logits = buildSoftMax(h_relu,n_fc,classes)

    return y_logits


if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='matlabexports/imdb_5indiv_3000_1000_s1.mat', type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--num_train', default = 1, type = int)
    parser.add_argument('--num_val', default = 107, type = int)
    parser.add_argument('--num_test', default = 0, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--itsPerEpoch', default = 10, type = int)
    args = parser.parse_args()

    # path = args.dataset
    num_train = args.num_train
    num_test = args.num_test
    num_valid = args.num_val

    Y_train, X_train, Y_valid, X_valid, Y_test, X_test = loadData(num_train, num_valid, num_test)
    # resolution = np.prod(imsize)

    # fakeDoc = Y_train[:,1]
    # fakeDoc[fakeDoc == 0] = -1
    # print X_train.shape
    # print fakeDoc.shape
    #
    #
    # X_train = np.vstack((fakeDoc, X_train.T))
    # X_train = X_train.T
    #
    # fakeDocVal = Y_valid[:,1]
    # fakeDocVal[fakeDocVal == 0] = -1
    # X_valid = np.vstack((fakeDocVal, X_valid.T))
    # X_valid = X_valid.T

    classes = 2
    x = tf.placeholder(tf.float32, [None, X_train.shape[1]])
    y = tf.placeholder(tf.float32, [None,2])
    keep_prob = tf.placeholder(tf.float32)

    y_logits, W_docs, b_docs = modelNoCorr(x,classes,keep_prob)


    # Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits,labels=y))
    # opt = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    opt = tf.train.GradientDescentOptimizer(0.01)
    optimizer = opt.minimize(cross_entropy)

    # Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    sensitivity, specificity, TP, FP, FN, TN = computeROCAccuracy(y, y_logits, name=None)

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

        indices = np.linspace(0, train_size, iter_per_epoch)
        indices = indices.astype('int')

        if args.train == 1:
            lossPlot = []
            accPlot = []
            sensPlot = []
            specPlot = []
            valAccPlot = []
            valLossPlot = []
            valSensPlot = []
            valSpecPlot = []

            # print "Start from:", start
            for epoch_i in range(n_epochs):
                lossEpoch = []
                accEpoch = []
                sensEpoch = []
                specEpoch = []

                for iter_i in range(iter_per_epoch-1):
                    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    WDocs, bDocs, loss, acc, sens, spec, TPtrain, FPtrain, FNtrain, TNtrain, yLogits = sess.run([W_docs, b_docs, cross_entropy,accuracy, sensitivity, specificity, TP, FP, FN, TN, y_logits],
                                    feed_dict={
                                        x: batch_xs,
                                        y: batch_ys,
                                        keep_prob: .8
                                    })
                    lossEpoch.append(loss)
                    accEpoch.append(acc)
                    sensEpoch.append(sens)
                    specEpoch.append(spec)

                    # featPlot.append(feat)
                    if iter_i % 1 == 0:

                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
                        print "TP " + str(TPtrain) + " FP " + str(FPtrain) + " FN " + str(FNtrain) + " TN " + str(TNtrain)
                        print yLogits
                        # convolve = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                        # print(convolve[0])
                    sess.run(optimizer, feed_dict={
                        x: batch_xs, y: batch_ys, keep_prob: 1})
                # convolve = sess.run(h_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0])

                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt",global_step=global_step)

                WDocsVal, bDocsVal, lossVal, accVal, sensVal, specVal = sess.run([W_docs, b_docs, cross_entropy,accuracy,sensitivity,specificity],
                             feed_dict={
                                 x: X_valid,
                                 y: Y_valid,
                                 keep_prob: 1.0
                             })

                # print "y ***********"
                # print batch_ys
                # print "logits ******"
                # print yLogits

                valAccPlot.append(accVal)
                valLossPlot.append(lossVal)
                valSensPlot.append(sensVal)
                valSpecPlot.append(specVal)
                print('(%d): ' % epoch_i +' Accuracy '+ str(accVal) +' sensitivity '+ str(sensVal) +' specificity '+ str(specVal))
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)
                lossPlot.append(np.mean(lossEpoch))
                accPlot.append(np.mean(accEpoch))
                sensPlot.append(np.mean(sensEpoch))
                specPlot.append(np.mean(specEpoch))
                # CNNplotter(lossPlot,accPlot,featPlot, Y_train,numIndiv)
                # CNNplotterROCFast(lossPlot,accPlot,sensPlot,specPlot, valAccPlot, valLossPlot, valSensPlot, valSpecPlot)
                CNNplotterROCFastWeights(WDocs, bDocs,WDocsVal,bDocsVal,lossPlot,accPlot,sensPlot,specPlot, valAccPlot, valLossPlot, valSensPlot, valSpecPlot)
                print "WDocs"
                print WDocs
                print "bDocs"
                print bDocs
                print "WDocsVal"
                print WDocsVal
                print "bDocsVal"
                print bDocsVal
                print 'Saving figure...'
                figname = ckpt_dir + '/result_' + str(epoch_i) + '.pdf'
                plt.savefig(figname)
        if args.train == 0:

            print('Accuracy for (%s): ' % ckpt.model_checkpoint_path + str(sess.run(accuracy,
                                                             feed_dict={
                                                                 x: X_test,
                                                                 y: Y_test,
                                                                 keep_prob: 1.0
                                                             })))
        if args.train == 2:
            print "Implement weight visualisation"
