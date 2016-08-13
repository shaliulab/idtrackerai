import os
import sys
sys.path.append('../utils')
import tensorflow as tf
from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
import argparse
import h5py
import numpy as np
from checkCheck import *
from pprint import *

def model(x,width, height, channels, classes, keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels],name='x_reshape')
    # conv1
    filter_size1 = 5
    n_filter1 = 15
    stride1 = [1,1,1,1]
    pad1 = 'SAME'
    h_conv1, w1, h1 = buildConv2D('Wconv1', 'bconv1', width, height, 1, x_tensor, filter_size1, n_filter1, stride1, pad1)
    # maxpool2d
    stride2 = [1,2,2,1]
    pool2 = 2
    pad2 = 'SAME'
    max_pool2, w2, h2 = maxpool2d('maxpool1',w1,h1, h_conv1, pool2,stride2,pad2)
    d2 = n_filter1
    # conv2
    filter_size3 = 5
    n_filter3 = 50
    stride3 = [1,1,1,1]
    pad3 = 'SAME'
    h_conv3, w3, h3 = buildConv2D('Wconv2', 'bconv2', w2, h2, d2, max_pool2, filter_size3, n_filter3, stride3, pad3)
    # maxpool2d
    stride4 = [1,2,2,1]
    pool4 = 2
    pad4 = 'SAME'
    max_pool4, w4, h4 = maxpool2d('maxpool2',w3,h3, h_conv3, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    h_conv5, w5, h5 = buildConv2D('Wconv3', 'bconv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
    d5 = n_filter5
    # linearize weights for fully-connected layer
    resolutionS = w5 * h5
    h_conv5_flat = tf.reshape(h_conv5, [-1, resolutionS*d5], name = 'h_conv5_reshape')
    # fully-connected 1
    n_fc = 100
    h_fc_drop = buildFc('Wfc1', 'bfc1', h_conv5_flat,w5,h5,d5,n_fc,keep_prob)
    h_relu = tf.nn.relu(h_fc_drop,name = 'lastRelu')
    y_logits = buildSoftMax('Wsoft1','bsoft1',h_relu,n_fc,classes)

    return y_logits, h_relu

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetTrain', default='36dpf_60indiv_22000ImPerInd_rotateAndCrop', type = str)
    parser.add_argument('--datasetTest', default=None, type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--loadCkpt_folder', default = "", type = str)
    parser.add_argument('--num_indiv', default = 60, type = int)
    parser.add_argument('--num_train', default = 22000, type = int)
    parser.add_argument('--num_test', default = 0, type = int)
    parser.add_argument('--num_ref', default = 0, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--batch_size', default = 250, type = int)
    args = parser.parse_args()

    pathTrain = args.datasetTrain
    pathTest = args.datasetTest
    num_indiv = args.num_indiv
    num_train = args.num_train
    num_test = args.num_test
    num_ref = args.num_ref
    ckpt_dir = args.ckpt_folder
    loadCkpt_folder = args.loadCkpt_folder

    # numIndiv, imsize, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = dataHelper0(path, num_train, num_test, num_valid, ckpt_dir)

    print "\n****** Loading database ******\n"
    numIndiv, imsize, \
    X_train, Y_train, \
    X_val, Y_val, \
    X_test, Y_test, \
    X_ref, Y_ref = loadDataBase(pathTrain, num_indiv, num_train, num_test, num_ref, ckpt_dir,pathTest)

    print '\n train size:    images  labels'
    print X_train.shape, Y_train.shape
    print 'val size:    images  labels'
    print X_val.shape, Y_val.shape
    print 'test size:    images  labels'
    print X_test.shape, Y_test.shape
    print 'ref size:    images  labels'
    print X_ref.shape, Y_ref.shape

    resolution = np.prod(imsize)
    classes = numIndiv

    x = tf.placeholder(tf.float32, [None, resolution], name = 'images')
    y = tf.placeholder(tf.float32, [None, classes], name = 'labels')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    y_logits, h_relu = model(x,imsize[1],imsize[2],imsize[0],classes,keep_prob)


    # Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y, name = 'CrossEntropy'), name = 'CrossEntropyMean')
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(cross_entropy, name = 'MinimizedFunc')
    optimizer = tf.train.GradientDescentOptimizer(0.01, name = 'OptMethod').minimize(cross_entropy, name = 'MinimizedFunc')

    # Monitor accuracy
    prediction = tf.argmax(y_logits, 1,name='prediction')
    truth = tf.argmax(y,1,name='truth')
    correct_prediction = tf.equal(prediction, truth,name='correctPrediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'),name='overallAccuracy')

    # Create counter for epochs and savers
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver_model = createSaver('soft', False, 'saver_model')
    saver_softmax = createSaver('soft', True, 'saver_softmax')


    print "\n****** Entering TF session ******\n"

    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()
        # tf.train.write_graph(sess.graph_def,ckpt_dir, 'train.pb',as_text=False)

        '''
        ************************************************************************
        *******************************Training*********************************
        ************************************************************************
        '''
        if args.train == 1:
            print "\n****** Starting training session ******\n"
            # Create folder for checkpoint if does not exist
            print "\n****** Checking Folders (save/restore) ******\n"
            [ckpt_dir_model,ckpt_dir_softmax,ckpt_dir_figures] = createCkptFolder( ckpt_dir, ['model', 'softmax', 'figures'])

            # load state of the training from the checkpoint
            print "\n"
            restoreFromFolder(ckpt_dir_model, saver_model, sess)
            restoreFromFolder(ckpt_dir_softmax, saver_softmax, sess)

            # counter for epochs
            start = global_step.eval() # get last global_step
            print "\nStart from:", start

            # We'll now train in minibatches and report accuracy, loss:
            n_epochs = args.num_epochs - start
            batch_size = args.batch_size
            train_size = len(Y_train)
            iter_per_epoch = int(np.ceil(np.true_divide(train_size,batch_size)))

            val_size = len(Y_val)
            val_iter_per_epoch = int(np.ceil(np.true_divide(val_size,batch_size)))

            print "Train size:", train_size
            print "Batch size:", batch_size
            print "Iter per epoch:", iter_per_epoch
            print "validation's batches"
            print "val size:", val_size
            print "val Iter per epoch:", val_iter_per_epoch

            indices = np.linspace(0, train_size, iter_per_epoch)
            indices = indices.astype('int')
            Vindices = np.linspace(0, val_size, val_iter_per_epoch)
            Vindices = Vindices.astype('int')

            if start == 0 or not os.path.exists(ckpt_dir_model + "/lossAcc.pkl"):
                lossPlot = []
                accPlot = []
                valLossPlot = []
                valAccPlot = []
                indivAccPlot = []
                indivValAccPlot = []
            else:
                ''' load from pickle '''
                print 'Loading loss and accuracies from previous checkpoint...'
                lossAccDict = pickle.load( open( ckpt_dir_model + "/lossAcc.pkl", "rb" ) )

                lossPlot = lossAccDict['loss']
                accPlot = lossAccDict['acc']
                valLossPlot = lossAccDict['valLoss']
                valAccPlot = lossAccDict['valAcc']
                indivAccPlot = lossAccDict['indivAcc']
                indivValAccPlot = lossAccDict['indivValAcc']

            # print "Start from:", start
            for epoch_i in range(n_epochs):
                lossEpoch = []
                accEpoch = []
                featPlot = []
                lossValEpoch = []
                accValEpoch = []
                indivAccEpoch = []
                valIIndivAccEpoch = []
                valFeatEpoch = []
                # refFeatEpoch = []

                '''
                **************************************
                training
                **************************************
                '''
                for iter_i in range(iter_per_epoch-1):
                    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    loss, acc, pred, tr, feat = sess.run([cross_entropy,accuracy, prediction, truth, h_relu],
                                    feed_dict={
                                        x: batch_xs,
                                        y: batch_ys,
                                        keep_prob: 1.0
                                    })
                    lossEpoch.append(loss)
                    accEpoch.append(acc)

                    # individual accuracy
                    indivPred = [np.true_divide(np.sum(np.logical_and(np.equal(pred, i), np.equal(tr, i)), axis=0), np.sum(np.equal(tr, i))) for i in range(classes)]
                    indivAccEpoch.append(indivPred)

                    if iter_i % round(np.true_divide(iter_per_epoch,4)) == 0:
                        indivPred = np.around(indivPred, decimals=2)
                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc) #+ ", Individual Accuracy= "
                        # pprint(indivPred)

                    sess.run(optimizer, feed_dict={
                        x: batch_xs, y: batch_ys, keep_prob: 1})
                # take the last batch to plot some features (from the last relu layer)
                featPlot = feat

                # nanmean because in minibatches some individuals could not appear...
                meanIndivAcc = np.nanmean(indivAccEpoch, axis=0)
                meanAcc = np.mean(meanIndivAcc)
                print('Training Accuracy (%d): ' % (start + epoch_i) + str(meanAcc) + " Individual Accuracy")
                pprint(meanIndivAcc)

                # update
                global_step.assign(global_step + 1).eval() # set and update(eval) global_step with index, i
                saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
                saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

                # dealing with massive validation sets (10% of train set), it is necessary to batch them
                '''
                **************************************
                validation
                **************************************
                '''
                for Viter_i in range(val_iter_per_epoch-1):
                    Vbatch_xs = X_val[Vindices[Viter_i]:Vindices[Viter_i+1]]
                    Vbatch_ys = Y_val[Vindices[Viter_i]:Vindices[Viter_i+1]]

                    valLoss, valAcc, valPred, valTr, Vfeat = sess.run([cross_entropy, accuracy, prediction, truth,h_relu],
                             feed_dict={
                                 x: Vbatch_xs,
                                 y: Vbatch_ys,
                                 keep_prob: 1.0
                             })
                    lossValEpoch.append(valLoss)
                    accValEpoch.append(valAcc)
                    valIndivPred = [np.true_divide(np.sum(np.logical_and(np.equal(valPred, i), np.equal(valTr, i)), axis=0), np.sum(np.equal(valTr, i))) for i in range(classes)]
                    valIIndivAccEpoch.append(valIndivPred)
                    valFeatEpoch.append(Vfeat)
                    if Viter_i % round(np.true_divide(val_iter_per_epoch,4)) == 0:
                        valIndivPred = np.around(valIndivPred, decimals=2)
                        print 'Validation Accuracy (batch = %d): ' % Viter_i + str(valAcc) #+ " Individual accuracy: "
                        # pprint(valIndivPred)

                valFeatEpoch = np.asarray(flatten(valFeatEpoch))
                meanValIndiviAcc = np.nanmean(valIIndivAccEpoch, axis=0)
                meanAcc = np.mean(meanValIndiviAcc)

                print 'Validation Accuracy (epoch = %d): ' % (start + epoch_i) + str(meanAcc) #+ "Individual Accuracy"
                pprint(meanValIndiviAcc)

                '''
                **************************************
                references
                **************************************
                '''
                # if Y_ref.shape[0] <= batch_size:
                #     print 'No need to batch references'
                #     Rfeat = sess.run(h_relu,
                #              feed_dict={
                #                  x: X_ref,
                #                  y: Y_ref,
                #                  keep_prob: 1.0
                #              })
                #     refFeatEpoch = Rfeat
                # else:
                #     # print Rindices
                #     # print ref_iter_per_epoch
                #     for Riter_i in range(ref_iter_per_epoch-1):
                #         Rbatch_xs = X_ref[Rindices[Riter_i]:Rindices[Riter_i+1]]
                #         Rbatch_ys = Y_ref[Rindices[Riter_i]:Rindices[Riter_i+1]]
                #
                #         Rfeat = sess.run(h_relu,
                #                  feed_dict={
                #                      x: Rbatch_xs,
                #                      y: Rbatch_ys,
                #                      keep_prob: 1.0
                #                  })
                #         refFeatEpoch.append(Rfeat)
                #     refFeatEpoch = np.asarray(flatten(refFeatEpoch))

                '''
                **************************************
                references accuracy
                **************************************
                '''
                # # idP, overallAccRef, indivAccRef = computeRefAccuracy(refFeatEpoch,Y_ref,valFeatEpoch,Y_val)
                # # print 'Validation Accuracy with Ref (epoch = %d): ' % (start + epoch_i) + str(overallAccRef) + ', individual accuracy = '
                # # pprint(indivAccRef)
                #
                '''
                **************************************
                saving dict with loss function and accuracy values
                **************************************
                '''
                lossPlot.append(np.mean(lossEpoch))
                accPlot.append(np.mean(accEpoch))
                valLossPlot.append(np.mean(lossValEpoch))
                valAccPlot.append(np.mean(accValEpoch))
                indivAccPlot.append(meanIndivAcc)
                indivValAccPlot.append(meanValIndiviAcc)

                lossSpeed, lossAccel = computeDerivatives(lossPlot)
                accSpeed, accAccel = computeDerivatives(accPlot)
                valLossSpeed, valLossAccel = computeDerivatives(valLossPlot)
                valAccSpeed, valAccAccel = computeDerivatives(valAccPlot)

                lossAccDict = {
                    'loss': lossPlot,
                    'valLoss': valLossPlot,
                    'lossSpeed': lossSpeed,
                    'valLossSpeed': valLossSpeed,
                    'lossAccel': lossAccel,
                    'valLossAccel': valLossAccel,
                    'acc': accPlot,
                    'valAcc': valAccPlot,
                    'accSpeed': accSpeed,
                    'valAccSpeed': valAccSpeed,
                    'accAccel': accAccel,
                    'valAccAccel': valAccAccel,
                    'indivAcc': indivAccPlot,
                    'indivValAcc': indivValAccPlot,
                    # 'indivValAccRef': indivAccRef,
                    'features': featPlot,
                    'labels': one_hot_to_dense(batch_ys) # labels of the last batch to plot some features
                    }

                pickle.dump( lossAccDict , open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )

                '''
                *******************
                Plotter
                *******************
                '''
                # CNNplotterFast(lossPlot,accPlot,valAccPlot,valLossPlot,meanIndivAcc,meanValIndiviAcc)
                CNNplotterFast(lossAccDict)
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)

                print 'Saving figure...'
                figname = ckpt_dir + '/figures/result_' + str(epoch_i) + '.pdf'
                plt.savefig(figname)
                print '-------------------------------'

        '''
        *************************************************************************
        ********************************Testing*********************************
        ************************************************************************
        '''
        if args.train == 0:

            print "\n****** Starting testing session ******\n"

            print "\n****** Checking Folders (save/restore) ******\n"

            # Create ckpt folder for the test if does not exist,
            # if it exist it gives the path to the cktp subfolders
            ckpt_dir = ckpt_dir + '_test'
            [ckpt_dir_model,ckpt_dir_softmax,ckpt_dir_figures] = createCkptFolder(ckpt_dir, ['model', 'softmax', 'figures'])

            # Load weights from a pretrained model if there is not any model saved
            # in the ckpt folder of the test
            ckpt = tf.train.get_checkpoint_state(ckpt_dir_model)
            if not (ckpt and ckpt.model_checkpoint_path):
                if loadCkpt_folder:
                    loadCkpt_folder = loadCkpt_folder + '/model'
                    print 'loading weigths from ' + loadCkpt_folder
                    restoreFromFolder(loadCkpt_folder, saver_model, sess)
                    global_step.assign(0).eval()

                else:
                    raise NameError('It is not possible to perform knowledge transfer, give a folder containing a trained model')
            else:
                restoreFromFolder(ckpt_dir_model, saver_model, sess)
                restoreFromFolder(ckpt_dir_softmax, saver_softmax, sess)


            start = global_step.eval()
            print "\nStart from:", start

            ''' ****** fine tuning ******'''
            # We'll now fine tune in minibatches and report accuracy, loss:
            n_epochs = args.num_epochs - start
            batch_size = args.batch_size

            ref_size = len(Y_ref)
            ref_iter_per_epoch = int(np.ceil(np.true_divide(ref_size,batch_size)))

            test_size = len(Y_test)
            test_iter_per_epoch = int(np.ceil(np.true_divide(test_size,batch_size)))

            print "Batch size:", batch_size
            print "References's batches"
            print "Ref size:", ref_size
            print "Ref iter per epoch:", ref_iter_per_epoch
            print "test's batches"
            print "test size:", test_size
            print "test Iter per epoch:", test_iter_per_epoch

            Rindices = np.linspace(0, ref_size, ref_iter_per_epoch)
            Rindices = Rindices.astype('int')
            Tindices = np.linspace(0, test_size, test_iter_per_epoch)
            Tindices = Tindices.astype('int')

            if start == 0 or not os.path.exists(ckpt_dir_model + "/lossAcc.pkl"):
                # ref lists for plotting
                refLossPlot = []
                refAccPlot = []
                refIndivAccPlot = []
                # test lists for ploting
                testLossPlot = []
                testAccPlot = []
                testIndivAccPlot = []
            else:
                ''' load from pickle '''
                print 'Loading loss and accuracies from previous checkpoint...'
                lossAccDict = pickle.load( open( ckpt_dir_model + "/lossAcc.pkl", "rb" ) )
                # ref lists for plotting
                refLossPlot = lossAccDict['loss']
                refAccPlot = lossAccDict['acc']
                refIndivAccPlot = lossAccDict['indivAcc']
                # test lists for plotting
                testLossPlot = lossAccDict['valLoss']
                testAccPlot = lossAccDict['valAcc']
                testIndivAccPlot = lossAccDict['indivValAcc']

            # print "Start from:", start
            for epoch_i in range(n_epochs):
                print '**** Epoch %i ****' % epoch_i
                '''
                **************************************
                Fine tuning
                **************************************
                '''
                if Y_ref.shape[0] <= batch_size:
                    # Run forward step to compute loss, accuracy...
                    refLoss, refAcc, refPred, refTr, refFeat = sess.run([cross_entropy,accuracy, prediction, truth, h_relu],
                             feed_dict={
                                 x: X_ref,
                                 y: Y_ref,
                                 keep_prob: 1.0
                             })

                    # individual accuracy
                    refIndivAcc = [np.true_divide(np.sum(np.logical_and(np.equal(refPred, i), np.equal(refTr, i)), axis=0), np.sum(np.equal(refTr, i))) for i in range(classes)]
                    refMeanIndivAcc = np.mean(refIndivAcc)

                    # Run backward step to compute and apply the gradients
                    sess.run(optimizer, feed_dict={
                        x: X_ref, y: Y_ref, keep_prob: 1})

                    # Labels to plot the features
                    refFeatLabels = Y_ref

                else:
                    print 'Fine tunning with batches of size %i' % batch_size
                    # lists to save each batch of the epoch
                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []

                    for iter_i in range(ref_iter_per_epoch-1):
                        # Take data for the batch
                        batch_xs = X_ref[Rindices[iter_i]:Rindices[iter_i+1]]
                        batch_ys = Y_ref[Rindices[iter_i]:Rindices[iter_i+1]]

                        # Run forward step to compute loss, accuracy...
                        batchLoss, batchAcc, batchPred, batchTr, batchFeat = sess.run([cross_entropy,accuracy, prediction, truth, h_relu],
                                        feed_dict={
                                            x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        # individual accuracy
                        indivBatchAcc = [np.true_divide(np.sum(np.logical_and(np.equal(batchPred, i), np.equal(batchTr, i)), axis=0), np.sum(np.equal(batchTr, i))) for i in range(classes)]

                        # Append batch results to the lists for the epoch
                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

                        # Print per batch loss and accuracies
                        if iter_i % round(np.true_divide(ref_iter_per_epoch,4)) == 0:
                            print "Batch " + str(iter_i) + \
                                ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                                ", Training Accuracy= " + "{:.5f}".format(batchAcc)

                        # Run backward step to compute and apply the gradients
                        sess.run(optimizer, feed_dict={
                            x: batch_xs, y: batch_ys, keep_prob: 1.0})

                    # When fine tuning with batches we take the features of the last batch
                    # as the features to be ploted
                    refFeat = batchFeat
                    refFeatLabels = batch_ys

                    # Compute mean loss, acc, and indivAcc for the epoch
                    # Note that we are overwriting the loss and acc of the last batch
                    # we do this to be consistent with the names of the case in which there are not batches
                    # This way the appending of results for the plot can be the same for the case with or
                    # without batches.
                    refLoss = np.mean(lossEpoch)
                    refAcc = np.mean(accEpoch)
                    refIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                # Batch finished
                print('Fine tunning (epoch %d): ' % (start + epoch_i) + \
                    " Loss=" + "{:.6f}".format(refLoss) + \
                    ", Accuracy=" + "{:.5f}".format(refAcc) + \
                    ", Individual Accuracy=")
                print(refIndivAcc)

                # update global step and save model
                global_step.assign(global_step + 1).eval() # set and update(eval) global_step with index, i
                saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
                saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

                '''
                **************************************
                Testing
                **************************************
                '''
                if Y_test.shape[0] <= batch_size:
                    print 'Testing'
                    # Run forward step to compute loss, accuracy...
                    testLoss, testAcc, testPred, testTr, testFeat = sess.run([cross_entropy,accuracy, prediction, truth, h_relu],
                             feed_dict={
                                 x: X_test,
                                 y: Y_test,
                                 keep_prob: 1.0
                             })

                    # individual accuracy
                    testIndivAcc = [np.true_divide(np.sum(np.logical_and(np.equal(testPred, i), np.equal(testTr, i)), axis=0), np.sum(np.equal(testTr, i))) for i in range(classes)]

                else:
                    print 'Testing with batches of size %i' % batch_size
                    # lists to save each batch of the epoch
                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []

                    for iter_i in range(test_iter_per_epoch-1):
                        # Take data for the batch
                        batch_xs = X_test[Tindices[iter_i]:Tindices[iter_i+1]]
                        batch_ys = Y_test[Tindices[iter_i]:Tindices[iter_i+1]]

                        # Run forward step to compute loss, accuracy...
                        batchLoss, batchAcc, batchPred, batchTr, batchFeat = sess.run([cross_entropy,accuracy, prediction, truth, h_relu],
                                        feed_dict={
                                            x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        # individual accuracy
                        indivBatchAcc = [np.true_divide(np.sum(np.logical_and(np.equal(batchPred, i), np.equal(batchTr, i)), axis=0), np.sum(np.equal(batchTr, i))) for i in range(classes)]

                        # Append batch results to the lists for the epoch
                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

                        # Print per batch loss and accuracies
                        if iter_i % round(np.true_divide(test_iter_per_epoch,4)) == 0:
                            print "Batch " + str(iter_i) + \
                                ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                                ", Training Accuracy= " + "{:.5f}".format(batchAcc)

                    # Compute mean loss, acc, and indivAcc for the epoch
                    # Note that we are overwriting the loss and acc of the last batch
                    # we do this to be consistent with the names of the case in which there are not batches
                    # This way the appending of results for the plot can be the same for the case with or
                    # without batches.
                    testLoss = np.mean(lossEpoch)
                    testAcc = np.mean(accEpoch)
                    testIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                # Batch finished
                print('Testing (epoch %d): ' % (start + epoch_i) + \
                    " Loss=" + "{:.6f}".format(testLoss) + \
                    ", Accuracy=" + "{:.5f}".format(testAcc) + \
                    ", Individual Accuracy=")
                print(testIndivAcc)

                '''
                **************************************
                saving dict with loss function and accuracy values
                **************************************
                '''

                # References
                refLossPlot.append(refLoss)
                refAccPlot.append(refAcc)
                refIndivAccPlot.append(refIndivAcc)
                refLossSpeed, refLossAccel = computeDerivatives(refLossPlot)
                refAccSpeed, refAccAccel = computeDerivatives(refAccPlot)
                refFeatPlot = refFeat

                # Test
                testLossPlot.append(testLoss)
                testAccPlot.append(testAcc)
                testIndivAccPlot.append(testIndivAcc)
                testLossSpeed, testLossAccel = computeDerivatives(testLossPlot)
                testAccSpeed, testAccAccel = computeDerivatives(testAccPlot)

                lossAccDict = {
                    'loss': refLossPlot,
                    'valLoss': testLossPlot,
                    'lossSpeed': refLossSpeed,
                    'valLossSpeed': testLossSpeed,
                    'lossAccel': refLossAccel,
                    'valLossAccel': testLossAccel,
                    'acc': refAccPlot,
                    'valAcc': testAccPlot,
                    'accSpeed': refAccSpeed,
                    'valAccSpeed': testAccSpeed,
                    'accAccel': refAccAccel,
                    'valAccAccel': testAccAccel,
                    'indivAcc': refIndivAccPlot,
                    'indivValAcc': testIndivAccPlot,
                    # 'indivValAccRef': indivAccRef,
                    'features': refFeatPlot,
                    'labels': one_hot_to_dense(refFeatLabels) # labels of the last batch of the references to plot some features
                    }

                pickle.dump( lossAccDict , open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )

                '''
                *******************
                Plotter
                *******************
                '''
                # CNNplotterFast(lossPlot,accPlot,valAccPlot,valLossPlot,meanIndivAcc,meanValIndiviAcc)
                CNNplotterFast(lossAccDict)
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)

                print 'Saving figure...'
                figname = ckpt_dir + '/figures/result_' + str(epoch_i) + '.pdf'
                plt.savefig(figname)
                print '-------------------------------'

        '''
        ************************************************************************
        ************************************************************************
        ************************************************************************
        ************************************************************************
        ************************************************************************
        ************************************************************************
        '''
        if args.train == 2:

            ckpt_model = tf.train.get_checkpoint_state(ckpt_dir_model)
            ckpt_softmax = tf.train.get_checkpoint_state(ckpt_dir_softmax)

            print ckpt_model
            print ckpt_softmax

            if ckpt_model and ckpt_model.model_checkpoint_path:
                print ckpt_model.model_checkpoint_path
                print_tensors_in_checkpoint_file(ckpt_model.model_checkpoint_path, "")
            if ckpt_softmax and ckpt_softmax.model_checkpoint_path:
                print ckpt_softmax.model_checkpoint_path
                print_tensors_in_checkpoint_file(ckpt_softmax.model_checkpoint_path, "")
