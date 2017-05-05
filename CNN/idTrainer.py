# Import standard libraries
import os
import sys
import numpy as np
import warnings
import time

# Import third party libraries
import tensorflow as tf

# Import application/library specifics
sys.path.append('IdTrackerDeep/utils')

from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
from cnn_architectures import *
from py_utils import *

def _add_loss_summary(loss):
    tf.summary.scalar(loss.op.name, loss)

def loss(y,y_logits):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y, name = 'CrossEntropy'), name = 'CrossEntropyMean')
    _add_loss_summary(cross_entropy)
    return cross_entropy

def optimize(loss,lr,varToTrain=[],use_adam = False):

    if not use_adam:
        print 'Training with SGD'
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif use_adam:
        print 'Training with ADAM'
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if not varToTrain:
        train_op = optimizer.minimize(loss=loss)
    else:
        train_op = optimizer.minimize(loss,var_list = varToTrain)
    return train_op, global_step

# def optimizeSoftmax(loss,lr,softVariables):
#     optimizer = tf.train.GradientDescentOptimizer(lr)
#     # optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
#     global_step = tf.Variable(0, name='global_step', trainable=False)
#     train_op = optimizer.minimize(loss,var_list = softVariables)
#     return train_op, global_step

def evaluation(y,y_logits,classes):
    accuracy, indivAcc = individualAccuracy(y,y_logits,classes)
    return accuracy, indivAcc

def placeholder_inputs(batch_size, width, height, channels, classes):
    images_placeholder = tf.placeholder(tf.float32, [None, width, height, channels], name = 'images')
    labels_placeholder = tf.placeholder(tf.float32, [None, classes], name = 'labels')

    return images_placeholder, labels_placeholder

def get_batch_indices(numImages,batch_size):
    indices = range(0,numImages,batch_size)# np.linspace(0, numImages, iter_per_epoch)
    indices.append(numImages)
    indices = np.asarray(indices)
    iter_per_epoch = indices.shape[0]-1
    indices = indices.astype('int')

    return indices, iter_per_epoch

def get_batch(batchNum, iter_per_epoch, indices, images_pl, labels_pl, keep_prob_pl, images, labels, keep_prob):
    if iter_per_epoch > 1:
        images_feed = images[indices[batchNum]:indices[batchNum+1]]
        labels_feed = labels[indices[batchNum]:indices[batchNum+1]]
    else:
        images_feed = images
        labels_feed = labels

    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_prob_pl: keep_prob
    }
    return feed_dict

def run_batch(sess, opsList, indices, batchNum, iter_per_epoch, images_pl,  labels_pl, keep_prob_pl, images, labels, keep_prob):
    feed_dict = get_batch(batchNum, iter_per_epoch, indices, images_pl, labels_pl, keep_prob_pl, images, labels, keep_prob)
    # Run forward step to compute loss, accuracy...
    outList = sess.run(opsList, feed_dict=feed_dict)
    outList.append(feed_dict)

    return outList

def run_training(X_t, Y_t, X_v, Y_v, X_test, Y_test,
    width, height, channels, classes,
    ckpt_dir, loadCkpt_folder,batch_size, num_epochs,
    Tindices, Titer_per_epoch,
    Vindices, Viter_per_epoch,
    TestIndices, TestIter_per_epoch,
    keep_prob = 1.0,lr = 0.01,
    printFlag=True, checkLearningFlag = False,
    onlySoftmax=False, onlyFullyConnected = False,
    saveFlag = True,
    use_adam = False):

    with tf.Graph().as_default():
        images_pl, labels_pl = placeholder_inputs(batch_size, width, height, channels, classes)
        keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

        logits, relu, (W1,W3,W5), softVar = inference1(images_pl, width, height, channels, classes, keep_prob_pl)

        cross_entropy = loss(labels_pl,logits)

        varToTrain = []
        if onlyFullyConnected:
            print '********************************************************'
            print 'We will only train the softmax and fully connected...'
            print '********************************************************'
            with tf.variable_scope("fully-connected1", reuse=True) as scope:
                FcW = tf.get_variable("weights")
                FcB = tf.get_variable("biases")
            print 'Fc W variables, ', FcW
            print 'Fc B variables, ', FcB
            varToTrain.append([FcW, FcB])
            with tf.variable_scope("softmax1", reuse=True) as scope:
                softW = tf.get_variable("weights")
                softB = tf.get_variable("biases")
            print 'softmax W variables, ', softW
            print 'softmax B variables, ', softB
            varToTrain.append([softW,softB])
        elif onlySoftmax:
            print '********************************************************'
            print 'We will only train the softmax...'
            print '********************************************************'
            with tf.variable_scope("softmax1", reuse=True) as scope:
                softW = tf.get_variable("weights")
                softB = tf.get_variable("biases")
            print 'softmax W variables, ', softW
            print 'softmax B variables, ', softB
            varToTrain.append([softW,softB])
        else:
            print '********************************************************'
            print 'We will only train the whole network...'
            print '********************************************************'

        varToTrain = flatten(varToTrain)

        train_op, global_step = optimize(cross_entropy,lr,varToTrain,use_adam)

        accuracy, indivAcc = evaluation(labels_pl,logits,classes)

        summary_op = tf.summary.merge_all()

        saver_model = createSaver('soft', False, 'saver_model')
        saver_softmax = createSaver('soft', True, 'saver_softmax')

        with tf.Session() as sess:
            # you need to initialize all variables
            tf.initialize_all_variables().run()

            print "\n****** Starting training session ******\n"
            # Create folder for checkpoint if does not exist
            print "\n****** Checking Folders (save/restore) ******\n"
            [ckpt_dir_model,ckpt_dir_softmax,ckpt_dir_figures] = createCkptFolder( ckpt_dir, ['model', 'softmax', 'figures'])

            # Load weights from a pretrained model if there is not any model saved
            # in the ckpt folder of the test
            ckpt = tf.train.get_checkpoint_state(ckpt_dir_model)
            print "************************************************************"
            print ckpt_dir_model
            print ckpt
            print loadCkpt_folder
            print "************************************************************"
            if not (ckpt and ckpt.model_checkpoint_path):
                if loadCkpt_folder:
                    print '********************************************************'
                    print 'We are only loading the model'
                    print '********************************************************'
                    loadCkpt_folder = loadCkpt_folder + '/model'
                    print 'loading weigths from ' + loadCkpt_folder
                    restoreFromFolder(loadCkpt_folder, saver_model, sess)
                    global_step.assign(0).eval()

                else:
                    warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')
            else:
                print "\n"
                print '********************************************************'
                print 'Shit, we are also loading the softmax'
                print '********************************************************'
                restoreFromFolder(ckpt_dir_model, saver_model, sess)
                restoreFromFolder(ckpt_dir_softmax, saver_softmax, sess)

            # counter for epochs
            start = global_step.eval() # get last global_step
            print "\nStart from:", start
            # We'll now fine tune in minibatches and report accuracy, loss:
            n_epochs = num_epochs - start
            print 'number of epochs to train, ', n_epochs

            summary_writerT = tf.summary.FileWriter(ckpt_dir + '/train',sess.graph)
            summary_writerV = tf.summary.FileWriter(ckpt_dir + '/val',sess.graph)

            print os.path.exists(ckpt_dir_model + "/lossAcc.pkl")
            print start
            if start == 0 or not os.path.exists(ckpt_dir_model + "/lossAcc.pkl"):
                print 'Initiallizing lists to save loss, acc and indivAcc'
                start = 0
                print "\nStart from:", start
                n_epochs = num_epochs - start
                # train lists for plotting
                trainLossPlot = []
                trainAccPlot = []
                trainIndivAccPlot = []
                # val lists for ploting
                valLossPlot = []
                valAccPlot = []
                valIndivAccPlot = []

                # test lists for ploting
                testLossPlot = []
                testAccPlot = []
                testIndivAccPlot = []
                # time for each epoch
                epochTime = []
            else:
                ''' load from pickle '''
                print 'Loading lists to save loss, acc and indivAcc'
                lossAccDict = pickle.load( open( ckpt_dir_model + "/lossAcc.pkl", "rb" ) )
                # train lists for plotting
                trainLossPlot = lossAccDict['loss']
                trainAccPlot = lossAccDict['acc']
                trainIndivAccPlot = lossAccDict['indivAcc']
                # val lists for plotting
                valLossPlot = lossAccDict['valLoss']
                valAccPlot = lossAccDict['valAcc']
                valIndivAccPlot = lossAccDict['indivValAcc']
                valIndivAcc = valIndivAccPlot[-1]
                # test lists for plotting
                testLossPlot = lossAccDict['testLoss']
                testAccPlot = lossAccDict['testAcc']
                testIndivAccPlot = lossAccDict['indivTestAcc']

                epochTime = lossAccDict['epochTime']

            # print "Start from:", start
            opListTrain = [train_op, cross_entropy, accuracy, indivAcc, relu, W1,W3,W5, softVar[0],logits]
            opListVal = [cross_entropy, accuracy, indivAcc, relu]

            stored_exception = None
            epoch_i = 0
            overfittingCounter = 0
            overfittingCounterTh = 5
            # WConv1old = 0.
            # WConv3old = 0.
            # WConv5old = 0.
            # softWold = 0.
            while epoch_i <= n_epochs:
                t0 = time.time()
                minNumEpochsCheckLoss = 10
                if start + epoch_i > 1:
                    if start + epoch_i > minNumEpochsCheckLoss: #and start > 100:
                        float_info = sys.float_info
                        minFloat = float_info[3]
                        # print 'valLossPlot, ', valLossPlot
                        currLoss = valLossPlot[-1]
                        prevLoss = valLossPlot[-minNumEpochsCheckLoss]
                        if currLoss == 0.:
                            currLoss = minFloat
                        if prevLoss == 0.:
            			    prevLoss = minFloat
                        print 'currLoss, ', currLoss
                        print 'prevLoss, ', prevLoss
                        if np.isnan(currLoss):
                            if printFlag:
                                print '\nThe validation loss is infinite, we stop the training'
                            break
                        if np.isnan(prevLoss):
                            if printFlag:
                                print '\nThe validation loss is infinite, we stop the training'
                            break
                        magCurr = int(np.log10(currLoss))-1
                        magPrev = int(np.log10(prevLoss))-1
                        epsilon = -.1*10**(magCurr)
                        epsilon2 = .01*10**(magCurr)

                        if printFlag:
                            print 'Losses difference (prev - curr) ', prevLoss-currLoss
                            print 'epsilon2 (if it is not changing much), ', epsilon2
                            print 'OverfittingCounter, ', overfittingCounter
                            print 'Threshold for overfittingCounter, ', overfittingCounterTh

                        if (prevLoss - currLoss) < 0:
                            overfittingCounter += 1
                            if printFlag:
                                print '\nOverfitting counter, ', overfittingCounter
                            if overfittingCounter >= overfittingCounterTh:
                                if printFlag:
                                    print '\n The network is overfitting, we stop the training'
                                    break
                        else:
                            overfittingCounter = 0

                        if (prevLoss - currLoss) < epsilon2 and checkLearningFlag:
                            if printFlag:
                                print '\nFinished, the network it is not learning more, we stop training'
                            break

                        if list(valIndivAcc) == list(np.ones(classes)):
                            if printFlag:
                                print '\nIndividual validations accuracy is 1 for all the animals'
                            break

                try:
                    epoch_counter = start + epoch_i
                    print '**** Epoch %i ****' % epoch_counter
                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []

                    ''' TRAINING '''
                    for iter_i in range(Titer_per_epoch):

                        _, batchLoss, batchAcc, indivBatchAcc, batchFeat, WConv1, WConv3, WConv5, softW,logitsS, feed_dict = run_batch(
                            sess, opListTrain, Tindices, iter_i, Titer_per_epoch,
                            images_pl, labels_pl, keep_prob_pl,
                            X_t, Y_t, keep_prob = keep_prob)

                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

                        # print '************************************************'
                        # print 'logits, ', logitsS
                        # print '************************************************'
                        # print 'Mean change of WConv1', np.mean(WConv1 - WConv1old)
                        # print 'Mean change of WConv3', np.mean(WConv3 - WConv3old)
                        # print 'Mean change of WConv5', np.mean(WConv5 - WConv5old)
                        # print 'Mean change of WSoft', np.mean(softW - softWold)
                        # WConv1old = WConv1
                        # WConv3old = WConv3
                        # WConv5old = WConv5
                        # softWold = softW

                        # Print per batch loss and accuracies
                        if (Titer_per_epoch < 4 or iter_i % round(np.true_divide(Titer_per_epoch,4)) == 0):
                            print "Batch " + str(iter_i) + \
                                ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                                ", Training Accuracy= " + "{:.5f}".format(batchAcc)


                    trainFeat = batchFeat
                    trainFeatLabels = Y_t[Tindices[iter_i]:Tindices[iter_i+1]]

                    trainLoss = np.mean(lossEpoch)
                    trainAcc = np.mean(accEpoch)
                    trainIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                    # Batch finished
                    print('Train (epoch %d): ' % epoch_counter + \
                        " Loss=" + "{:.6f}".format(trainLoss) + \
                        ", Accuracy=" + "{:.5f}".format(trainAcc) + \
                        ", Individual Accuracy=")
                    print(trainIndivAcc)

                    # Summary writer
                    # summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    # summary_writerT.add_summary(summary_str, epoch_i)

                    # # update global step and save model
                    global_step.assign(global_step + 1).eval() # set and update(eval) global_step with index, i
                    # saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
                    # saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

                    ''' VALIDATION '''
                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []
                    for iter_i in range(Viter_per_epoch):

                        batchLoss, batchAcc, indivBatchAcc, batchFeat, feed_dict = run_batch(
                            sess, opListVal, Vindices, iter_i, Viter_per_epoch,
                            images_pl, labels_pl, keep_prob_pl,
                            X_v, Y_v, keep_prob = keep_prob)

                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

                        # Print per batch loss and accuracies
                        # if iter_i % round(np.true_divide(Viter_per_epoch,1)) == 0:
                        #     print "Batch " + str(iter_i) + \
                        #         ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                        #         ", Validation Accuracy= " + "{:.5f}".format(batchAcc)

                    valLoss = np.mean(lossEpoch)
                    valAcc = np.mean(accEpoch)
                    valIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                    # Batch finished

                    print('Validation (epoch %d): ' % epoch_counter + \
                        " Loss=" + "{:.6f}".format(valLoss) + \
                        ", Accuracy=" + "{:.5f}".format(valAcc) + \
                        ", Individual Accuracy=")
                    print(valIndivAcc)

                    # summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    # summary_writerV.add_summary(summary_str, epoch_i)

                    epochTime.append(time.time()-t0)
                    print 'Epoch time in seconds, ', epochTime[-1]

                    # ''' TEST '''
                    # lossEpoch = []
                    # accEpoch = []
                    # indivAccEpoch = []
                    # for iter_i in range(TestIter_per_epoch):
                    #
                    #     batchLoss, batchAcc, indivBatchAcc, batchFeat, feed_dict = run_batch(
                    #         sess, opListVal, TestIndices, iter_i, TestIter_per_epoch,
                    #         images_pl, labels_pl, keep_prob_pl,
                    #         X_test, Y_test, keep_prob = keep_prob)
                    #
                    #     lossEpoch.append(batchLoss)
                    #     accEpoch.append(batchAcc)
                    #     indivAccEpoch.append(indivBatchAcc)
                    #
                    #     # # Print per batch loss and accuracies
                    #     # if iter_i % round(np.true_divide(Viter_per_epoch,100)) == 0:
                    #     #     print "Batch " + str(iter_i) + \
                    #     #         ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                    #     #         ", Test Accuracy= " + "{:.5f}".format(batchAcc)
                    #
                    # testLoss = np.mean(lossEpoch)
                    # testAcc = np.mean(accEpoch)
                    # testIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                    # Batch finished

                    # print('Test (epoch %d): ' % epoch_counter + \
                    #     " Loss=" + "{:.6f}".format(testLoss) + \
                    #     ", Accuracy=" + "{:.5f}".format(testAcc) + \
                    #     ", Individual Accuracy=")
                    # print(testIndivAcc)

                    '''
                    **************************************
                    saving dict with loss function and accuracy values
                    **************************************
                    '''

                    # Train
                    trainLossPlot.append(trainLoss)
                    trainAccPlot.append(trainAcc)
                    trainIndivAccPlot.append(trainIndivAcc)
                    # trainLossSpeed, trainLossAccel = computeDerivatives(trainLossPlot)
                    # trainAccSpeed, trainAccAccel = computeDerivatives(trainAccPlot)
                    # trainFeatPlot = trainFeat

                    # Validations
                    valLossPlot.append(valLoss)
                    valAccPlot.append(valAcc)
                    valIndivAccPlot.append(valIndivAcc)
                    # valLossSpeed, valLossAccel = computeDerivatives(valLossPlot)
                    # valAccSpeed, valAccAccel = computeDerivatives(valAccPlot)

                    # Test
                    # testLossPlot.append(testLoss)
                    # testAccPlot.append(testAcc)
                    # testIndivAccPlot.append(testIndivAcc)
                    # testLossSpeed, testLossAccel = computeDerivatives(testLossPlot)
                    # testAccSpeed, testAccAccel = computeDerivatives(testAccPlot)

                    lossAccDict = {
                        'loss': trainLossPlot,
                        'valLoss': valLossPlot,
                        # 'testLoss': testLossPlot,
                        'acc': trainAccPlot,
                        'valAcc': valAccPlot,
                        # 'testAcc': testAccPlot,
                        'indivAcc': trainIndivAccPlot,
                        'indivValAcc': valIndivAccPlot,
                        # 'indivTestAcc': testIndivAccPlot,
                        'epochTime': epochTime
                        }

                    # weightsDict = {
                    #     'W1': WConv1,
                    #     'W3': WConv3,
                    #     'W5': WConv5
                    #     }

                    # # update global step and save model
                    # global_step.assign(global_step + 1).eval() # set and update(eval) global_step with index, i
                    # saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
                    # saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

                    # pickle.dump( lossAccDict, open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )
                    # pickle.dump( weightsDict, open( ckpt_dir_model + "/weightsDict.pkl", "wb" ) )
                    '''
                    *******************
                    Plotter
                    *******************
                    '''
                    ### uncomment to plot ----
                    # if epoch_i % 1 == 0:
                    #     CNNplotterFast2(lossAccDict, weightsDict)
                    #
                    #     print 'Saving figure...'
                    #     figname = ckpt_dir + '/figures/result_' + str(global_step.eval()) + '.pdf'
                    #     plt.savefig(figname)
                    # print '-------------------------------'
                    ### ---

                    if stored_exception:
                        break
                    epoch_i += 1
                except KeyboardInterrupt:
                    stored_exception = sys.exc_info()

            if stored_exception:
                raise stored_exception[0], stored_exception[1], stored_exception[2]

            ''' TEST '''
            lossEpoch = []
            accEpoch = []
            indivAccEpoch = []
            for iter_i in range(TestIter_per_epoch):

                batchLoss, batchAcc, indivBatchAcc, batchFeat, feed_dict = run_batch(
                    sess, opListVal, TestIndices, iter_i, TestIter_per_epoch,
                    images_pl, labels_pl, keep_prob_pl,
                    X_test, Y_test, keep_prob = keep_prob)

                lossEpoch.append(batchLoss)
                accEpoch.append(batchAcc)
                indivAccEpoch.append(indivBatchAcc)

                # # Print per batch loss and accuracies
                # if iter_i % round(np.true_divide(Viter_per_epoch,100)) == 0:
                #     print "Batch " + str(iter_i) + \
                #         ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                #         ", Test Accuracy= " + "{:.5f}".format(batchAcc)

            testLoss = np.mean(lossEpoch)
            testAcc = np.mean(accEpoch)
            testIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

            print('Test: ' + \
                " Loss=" + "{:.6f}".format(testLoss) + \
                ", Accuracy=" + "{:.5f}".format(testAcc) + \
                ", Individual Accuracy=")
            print(testIndivAcc)
            print 'number of fragments evaluated, ', len(accEpoch)
            print 'max accuracy, ', np.max(accEpoch), ', min accuracy, ', np.min(accEpoch)

            # Test
            testLossPlot.append(testLoss)
            testAccPlot.append(testAcc)
            testIndivAccPlot.append(testIndivAcc)


            lossAccDict['testLoss'] = testLossPlot
            lossAccDict['testAcc'] = accEpoch #
            lossAccDict['indivTestAcc'] = testIndivAccPlot


            # update global step and save model
            saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
            saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

    if saveFlag:
        print 'Saving lossAccDict...'
        pickle.dump( lossAccDict, open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )
        print 'lossAccDictSaved...'
    else:
        return lossAccDict, ckpt_dir_model
