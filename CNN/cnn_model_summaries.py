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
from cnn_architectures import *
import warnings

def _add_loss_summary(loss):
    tf.scalar_summary(loss.op.name, loss)

def loss(y,y_logits):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y, name = 'CrossEntropy'), name = 'CrossEntropyMean')
    _add_loss_summary(cross_entropy)
    return cross_entropy

def optimize(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss)
    return train_op, global_step

def evaluation(y,y_logits,classes):
    accuracy, indivAcc = individualAccuracy(y,y_logits,classes)
    return accuracy, indivAcc

def placeholder_inputs(batch_size, resolution):
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

def run_training(X_t, Y_t, X_v, Y_v, width, height, channels, classes, resolution, loadCkpt_folder, keep_prob = 1.0):
    with tf.Graph().as_default():
        images_pl, labels_pl = placeholder_inputs(batch_size, resolution)
        keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')

        logits, relu = inference1(images_pl, width, height, channels, classes, keep_prob_pl)

        cross_entropy = loss(labels_pl,logits)

        train_op, global_step =  optimize(cross_entropy)

        accuracy, indivAcc = evaluation(labels_pl,logits,classes)

        summary_op = tf.merge_all_summaries()

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
            if not (ckpt and ckpt.model_checkpoint_path):
                if loadCkpt_folder:
                    loadCkpt_folder = loadCkpt_folder + '/model'
                    print 'loading weigths from ' + loadCkpt_folder
                    restoreFromFolder(loadCkpt_folder, saver_model, sess)
                    global_step.assign(0).eval()

                else:
                    warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')
            else:
                print "\n"
                restoreFromFolder(ckpt_dir_model, saver_model, sess)
                restoreFromFolder(ckpt_dir_softmax, saver_softmax, sess)

            # counter for epochs
            start = global_step.eval() # get last global_step
            print "\nStart from:", start
            # We'll now fine tune in minibatches and report accuracy, loss:
            n_epochs = num_epochs - start

            summary_writerT = tf.train.SummaryWriter(ckpt_dir + '/train',sess.graph)
            summary_writerV = tf.train.SummaryWriter(ckpt_dir + '/val',sess.graph)

            if start == 0 or not os.path.exists(ckpt_dir_model + "/lossAcc.pkl"):
                # ref lists for plotting
                trainLossPlot = []
                trainAccPlot = []
                trainIndivAccPlot = []
                # test lists for ploting
                valLossPlot = []
                valAccPlot = []
                valIndivAccPlot = []
            else:
                ''' load from pickle '''
                lossAccDict = pickle.load( open( ckpt_dir_model + "/lossAcc.pkl", "rb" ) )
                # ref lists for plotting
                trainLossPlot = lossAccDict['loss']
                trainAccPlot = lossAccDict['acc']
                trainIndivAccPlot = lossAccDict['indivAcc']
                # test lists for plotting
                valLossPlot = lossAccDict['valLoss']
                valAccPlot = lossAccDict['valAcc']
                valIndivAccPlot = lossAccDict['indivValAcc']

            # print "Start from:", start
            opListTrain = [train_op, cross_entropy, accuracy, indivAcc, relu]
            opListVal = [cross_entropy, accuracy, indivAcc, relu]

            stored_exception = None
            epoch_i = 0
            while epoch_i <= n_epochs:
            # for epoch_i in range(n_epochs):
                try:
                    epoch_counter = start + epoch_i
                    print '**** Epoch %i ****' % epoch_counter
                    lossEpoch = []
                    accEpoch = []
                    indivAccEpoch = []

                    ''' TRAINING '''
                    for iter_i in range(Titer_per_epoch):

                        _, batchLoss, batchAcc, indivBatchAcc, batchFeat, feed_dict = run_batch(
                            sess, opListTrain, Tindices, iter_i, Titer_per_epoch,
                            images_pl, labels_pl, keep_prob_pl,
                            X_t, Y_t, keep_prob = keep_prob)

                        lossEpoch.append(batchLoss)
                        accEpoch.append(batchAcc)
                        indivAccEpoch.append(indivBatchAcc)

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
                        if iter_i % round(np.true_divide(Viter_per_epoch,4)) == 0:
                            print "Batch " + str(iter_i) + \
                                ", Minibatch Loss= " + "{:.6f}".format(batchLoss) + \
                                ", Training Accuracy= " + "{:.5f}".format(batchAcc)

                    valLoss = np.mean(lossEpoch)
                    valAcc = np.mean(accEpoch)
                    valIndivAcc = np.nanmean(indivAccEpoch, axis=0) # nanmean because in minibatches some individuals could not appear...

                    # Batch finished

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

                    pickle.dump( lossAccDict , open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )
                    '''
                    *******************
                    Plotter
                    *******************
                    '''
                    if epoch_i % 10 == 0:
                        CNNplotterFast(lossAccDict)

                        print 'Saving figure...'
                        figname = ckpt_dir + '/figures/result_' + str(global_step.eval()) + '.pdf'
                        plt.savefig(figname)
                    print '-------------------------------'

                    if stored_exception:
                        break
                    epoch_i += 1
                except KeyboardInterrupt:
                    stored_exception = sys.exc_info()

            if stored_exception:
                raise stored_exception[0], stored_exception[1], stored_exception[2]


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
batch_size = args.batch_size
num_epochs = args.num_epochs


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

    run_training(X_train, Y_train, X_val, Y_val, width, height, channels, classes, resolution, loadCkpt_folder)

if args.train == 0:
    numImagesT = Y_ref.shape[0]
    numImagesV = Y_test.shape[0]
    Tindices, Titer_per_epoch = get_batch_indices(numImagesT,batch_size)
    Vindices, Viter_per_epoch = get_batch_indices(numImagesV,batch_size)


    run_training(X_ref, Y_ref, X_test, Y_test, width, height, channels, classes, resolution, loadCkpt_folder, 0.5)
