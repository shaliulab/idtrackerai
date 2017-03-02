import os
import sys
sys.path.append('IdTrackerDeep/utils')

from tf_utils import *
from input_data_cnn import *
from cnn_utils import *
from cnn_architectures import *
from plotters import *

import tensorflow as tf
import argparse
import h5py
import numpy as np
# from checkCheck import *
from pprint import *
import warnings

def _add_loss_summary(loss):
    tf.summary.scalar(loss.op.name, loss)

def loss(y,y_logits):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_logits,labels=y, name = 'CrossEntropy'), name = 'CrossEntropyMean')
    _add_loss_summary(cross_entropy)
    return cross_entropy

def optimize(loss,lr):
    optimizer = tf.train.GradientDescentOptimizer(lr)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss)
    return train_op, global_step

def evaluation(y,y_logits,classes):
    accuracy, indivAcc = individualAccuracy(y,y_logits,classes)
    return accuracy, indivAcc

def placeholder_inputs(batch_size, resolution, classes):
    images_placeholder = tf.placeholder(tf.float32, [None, resolution], name = 'images')
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

def run_training(X_t, Y_t, X_v, Y_v, width, height, channels, classes, resolution, trainDict, accumDict, fragmentsDict, handlesDict, portraits, Tindices, Titer_per_epoch, Vindices, Viter_per_epoch, plotFlag = True, printFlag = True):

    # get data from trainDict
    loadCkpt_folder = trainDict['loadCkpt_folder']
    ckpt_dir = trainDict['ckpt_dir']
    fig_dir = trainDict['fig_dir']
    sessionPath = trainDict['sess_dir']
    num_epochs = trainDict['numEpochs']
    batch_size = trainDict['batchSize']
    lr = trainDict['lr']
    keep_prob = trainDict['keep_prob']
    lossAccDict = trainDict['lossAccDict']
    usedIndivIntervals = trainDict['usedIndivIntervals']
    idUsedIntervals = trainDict['idUsedIntervals']
    idUsedIndivIntervals = zip(usedIndivIntervals,idUsedIntervals)
    print '********************************************************************'
    print '********************************************************************'
    pprint(idUsedIndivIntervals)
    print '********************************************************************'
    print '********************************************************************'

    # get data from accumDict
    accumCounter = accumDict['counter']

    with tf.Graph().as_default():
        images_pl, labels_pl = placeholder_inputs(batch_size, resolution, classes)
        keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

        logits, relu,(W1,W3,W5) = inference1(images_pl, width, height, channels, classes, keep_prob_pl)

        cross_entropy = loss(labels_pl,logits)

        train_op, global_step =  optimize(cross_entropy,lr)

        accuracy, indivAcc = evaluation(labels_pl,logits,classes)

        summary_op = tf.summary.merge_all()

        saver_model = createSaver('soft', False, 'saver_model')
        saver_softmax = createSaver('soft', True, 'saver_softmax')

        with tf.Session() as sess:
            # you need to initialize all variables
            # tf.initialize_all_variables().run() #NOTE deprecated
            tf.global_variables_initializer().run()

            if printFlag:
                print "\n****** Starting training session ******\n"
                # Create folder for checkpoint if does not exist
                print "\n****** Checking Folders (save/restore) ******\n"

            [ckpt_dir_model,ckpt_dir_softmax] = createCkptFolder( ckpt_dir, ['model', 'softmax'])

            # Load weights from a pretrained model if there is not any model saved
            # in the ckpt folder of the test
            ckpt = tf.train.get_checkpoint_state(ckpt_dir_model)
            if (not (ckpt and ckpt.model_checkpoint_path)) and accumCounter == 0:
                if loadCkpt_folder:
                    if printFlag:
                        print '********************************************************'
                        print 'We are only loading the model'
                        print '********************************************************'
                    loadCkpt_folder = loadCkpt_folder + '/model'

                    if printFlag:
                        print 'loading weigths from ' + loadCkpt_folder

                    restoreFromFolder(loadCkpt_folder, saver_model, sess)
                    global_step.assign(0).eval()

                else:
                    warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')
            else:
                loadCkpt_folder_model = loadCkpt_folder + '/model'
                loadCkpt_folder_softmax = loadCkpt_folder + '/softmax'

                if printFlag:
                    print "\n"
                    print '********************************************************'
                    print 'We are also loading the softmax'
                    print '********************************************************'
                    print 'loading weigths from ' + loadCkpt_folder + '/model'
                    print 'loading softmax from ' + loadCkpt_folder + '/softmax'
                restoreFromFolder(loadCkpt_folder_model, saver_model, sess)
                restoreFromFolder(loadCkpt_folder_softmax, saver_softmax, sess)

            # counter for epochs
            start = global_step.eval() # get last global_step

            if printFlag:
                print "\nStart from:", start

            # We'll now fine tune in minibatches and report accuracy, loss:
            n_epochs = num_epochs - start

            summary_writerT = tf.summary.FileWriter(ckpt_dir + '/train',sess.graph)
            summary_writerV = tf.summary.FileWriter(ckpt_dir + '/val',sess.graph)

            if accumCounter == 0:
                # ref lists for plotting
                trainLossPlot = []
                trainAccPlot = []
                trainIndivAccPlot = []
                # test lists for ploting
                valLossPlot = []
                valAccPlot = []
                valIndivAccPlot = []
            else:
                # ref lists for plotting
                trainLossPlot = lossAccDict['loss']
                trainAccPlot = lossAccDict['acc']
                trainIndivAccPlot = lossAccDict['indivAcc']
                # test lists for plotting
                valLossPlot = lossAccDict['valLoss']
                valAccPlot = lossAccDict['valAcc']
                valIndivAccPlot = lossAccDict['indivValAcc']

            # print "Start from:", start
            # opListTrain = [train_op, cross_entropy, accuracy, indivAcc, relu]
            opListTrain = [train_op, cross_entropy, accuracy, indivAcc, relu, W1,W3,W5]
            opListVal = [cross_entropy, accuracy, indivAcc, relu]

            stored_exception = None
            epoch_i = 0
            valIndivAcc = np.zeros(classes)
            while epoch_i <= n_epochs:
                minNumEpochsCheckLoss = 10
                if start + epoch_i > 1:
                    if len(valLossPlot) > minNumEpochsCheckLoss + start: #and start > 100:
                        currLoss = valLossPlot[-1]
                        prevLoss = valLossPlot[-minNumEpochsCheckLoss]
                        magCurr = int(np.log10(currLoss))-1
                        magPrev = int(np.log10(prevLoss))-1
                        epsilon = -.1*10**(magCurr)
                        epsilon2 = .01*10**(magCurr)

                        if printFlag:
                            print 'Losses difference (prev - curr) ', prevLoss-currLoss
                            print 'epsilon (overfitting), ', epsilon
                            print 'epsilon2 (if it is not changing much), ', epsilon2

                        if np.mean(valIndivAcc) > .8: ###NOTE: decreased to .8 for large groups (38 animals)
                            if magCurr > magPrev:
                                if printFlag:
                                    print '\nOverfitting, passing to new set of images'

                                break
                            elif magCurr == magPrev:
                                if (prevLoss - currLoss) < epsilon:
                                    if printFlag:
                                        print '\nOverfitting, passing to new set of images'

                                    break
                            if (prevLoss - currLoss) < epsilon2:
                                if printFlag:
                                    print '\nFinished, passing to new set of images'

                                break
                            if list(valIndivAcc) == list(np.ones(classes)):
                                if printFlag:
                                    print '\nIndividual validations accuracy is 1 for all the animals'
                                break

                try:
                    epoch_counter = start + epoch_i

                    if printFlag:
                        print '\n**** Epoch %i ****' % epoch_counter

                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []

                    ''' TRAINING '''
                    for iter_i in range(Titer_per_epoch):

                        _, batchLoss, batchAcc, indivBatchAcc, batchFeat, WConv1, WConv3, WConv5, feed_dict = run_batch(
                            sess, opListTrain, Tindices, iter_i, Titer_per_epoch,
                            images_pl, labels_pl, keep_prob_pl,
                            X_t, Y_t, keep_prob = keep_prob)

                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

                        # Print per batch loss and accuracies
                        if printFlag:
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
                    if printFlag:
                        print('Train (epoch %d): ' % epoch_counter + \
                            " Loss=" + "{:.6f}".format(trainLoss) + \
                            ", Accuracy=" + "{:.5f}".format(trainAcc) + \
                            ", Individual Accuracy=")
                        print(trainIndivAcc)

                    # Summary writer
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writerT.add_summary(summary_str, epoch_i)

                    # update global step and save model
                    global_step.assign(global_step + 1).eval() # set and update(eval) global_step with index, i
                    saver_model.save(sess, ckpt_dir_model + "/model.ckpt", global_step = global_step)
                    saver_softmax.save(sess, ckpt_dir_softmax + "/softmax.ckpt",global_step = global_step)

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
                        #         ", Training Accuracy= " + "{:.5f}".format(batchAcc)

                    valLoss = np.mean(lossEpoch)
                    valAcc = np.mean(accEpoch)
                    valIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                    # Batch finished
                    if printFlag:
                        print('Validation (epoch %d): ' % epoch_counter + \
                            " Loss=" + "{:.6f}".format(valLoss) + \
                            ", Accuracy=" + "{:.5f}".format(valAcc) + \
                            ", Individual Accuracy=")
                        print(valIndivAcc)

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writerV.add_summary(summary_str, epoch_i)

                    '''
                    **************************************
                    saving dict with loss function and accuracy values
                    **************************************
                    '''

                    # References
                    trainLossPlot.append(trainLoss)
                    trainAccPlot.append(trainAcc)
                    trainIndivAccPlot.append(trainIndivAcc)
                    trainLossSpeed, trainLossAccel = computeDerivatives(trainLossPlot)
                    trainAccSpeed, trainAccAccel = computeDerivatives(trainAccPlot)
                    trainFeatPlot = trainFeat

                    # Test
                    valLossPlot.append(valLoss)
                    valAccPlot.append(valAcc)
                    valIndivAccPlot.append(valIndivAcc)
                    valLossSpeed, valLossAccel = computeDerivatives(valLossPlot)
                    valAccSpeed, valAccAccel = computeDerivatives(valAccPlot)

                    lossAccDict = {
                        'loss': trainLossPlot,
                        'valLoss': valLossPlot,
                        'lossSpeed': trainLossSpeed,
                        'valLossSpeed': valLossSpeed,
                        'lossAccel': trainLossAccel,
                        'valLossAccel': valLossAccel,
                        'acc': trainAccPlot,
                        'valAcc': valAccPlot,
                        'accSpeed': trainAccSpeed,
                        'valAccSpeed': valAccSpeed,
                        'accAccel': trainAccSpeed,
                        'valAccAccel': valAccSpeed,
                        'indivAcc': trainIndivAccPlot,
                        'indivValAcc': valIndivAccPlot,
                        # 'indivValAccRef': indivAccRef,
                        'features': trainFeatPlot,
                        'labels': one_hot_to_dense(trainFeatLabels) # labels of the last batch of the references to plot some features
                        }

                    weightsDict = {
                        'W1': WConv1,
                        'W3': WConv3,
                        'W5': WConv5
                        }

                    pickle.dump( lossAccDict, open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )
                    '''
                    *******************
                    Plotter
                    *******************
                    '''
                    ### uncomment to plot ----

                    if epoch_i % 1 == 0:
                        # CNNplotterFast2(lossAccDict, weightsDict)
                        # CNNplotterFast22(lossAccDict, weightsDict, idUsedIndivIntervals, accumDict,fragmentsDict,portraits,sessionPath, plotFlag)
                        handlesDict = CNNplotterFast22(epoch_counter,epoch_i,handlesDict,lossAccDict, idUsedIndivIntervals, accumDict,fragmentsDict,portraits,sessionPath)

                        print 'Saving figure...'
                        figname = fig_dir + '/result_' + str(global_step.eval()) + '.pdf'
                        plt.savefig(figname)
                    print '\n-------------------------------\n'
                    ### ---

                    if stored_exception:
                        break
                    epoch_i += 1
                except KeyboardInterrupt:
                    stored_exception = sys.exc_info()

            if stored_exception:
                raise stored_exception[0], stored_exception[1], stored_exception[2]

    pickle.dump( lossAccDict , open( sessionPath + "/lossAcc.pkl", "wb" ) )
    print 'You just saved the lossAccDict'
    trainDict['lossAccDict'] = lossAccDict
    return trainDict, handlesDict

"""
Sample calls:
Training:
python -i cnn_model_summaries.py
--ckpt_folder ckpt_Train_60indiv_36dpf_22000_transfer
--dataset_train 25dpf_60indiv_22000ImPerInd_rotateAndCrop

Trasnfer:
python -i cnn_model_summaries.py
--ckpt_folder ckpt_Train_60indiv_25dpf_25000_transfer
--load_ckpt_folder ckpt_dir_new
--dataset_train 25dpf_60indiv_26142ImPerInd_curvaturePortrait
"""

if __name__ == '__main__':
    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', default='36dpf_60indiv_29754ImPerInd_curvaturePortrait', type = str)
    # parser.add_argument('--dataset_train', default='36dpf_60indiv_22000ImPerInd_rotateAndCrop', type = str)
    parser.add_argument('--dataset_test', default=None, type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--load_ckpt_folder', default = "", type = str)
    parser.add_argument('--num_indiv', default = 60, type = int)
    parser.add_argument('--num_train', default = 25000, type = int)
    parser.add_argument('--num_test', default = 0, type = int)
    parser.add_argument('--num_ref', default = 0, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--batch_size', default = 250, type = int)
    parser.add_argument('--learning_rate', default = 0.001, type= float)
    args = parser.parse_args()

    pathTrain = args.dataset_train
    pathTest = args.dataset_test
    num_indiv = args.num_indiv
    num_train = args.num_train
    num_test = args.num_test
    num_ref = args.num_ref
    ckpt_dir = args.ckpt_folder
    loadCkpt_folder = args.load_ckpt_folder
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.learning_rate


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

    channels, width, height = imsize
    resolution = np.prod(imsize)
    classes = numIndiv

    '''
    ************************************************************************
    *******************************Training*********************************
    ************************************************************************
    '''
    if args.train == 1:
        numImagesT = Y_train.shape[0]
        numImagesV = Y_val.shape[0]
        Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)
        Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batch_size)

        run_training(X_train, Y_train, X_val, Y_val, width, height, channels, classes, resolution, ckpt_dir, loadCkpt_folder, accumCounter, batch_size, num_epochs, Tindices, Titer_per_epoch,
        Vindices, Viter_per_epoch,1.,lr)

    if args.train == 0:
        numImagesT = Y_ref.shape[0]
        numImagesV = Y_test.shape[0]
        Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)
        Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batch_size)


        run_training(X_ref, Y_ref, X_test, Y_test, width, height, channels, classes, resolution, ckpt_dir, loadCkpt_folder, batch_size, num_epochs, Tindices, Titer_per_epoch,
        Vindices, Viter_per_epoch, 0.5,lr)
