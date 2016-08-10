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


def model(x,y,width, height, channels, classes, keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels])
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
    max_pool2, w2, h2 = maxpool2d(w1,h1, h_conv1, pool2,stride2,pad2)
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
    max_pool4, w4, h4 = maxpool2d(w3,h3, h_conv3, pool4,stride4,pad4)
    d4 = n_filter3
    # conv4
    filter_size5 = 5
    n_filter5 = 100
    stride5 = [1,1,1,1]
    pad5 = 'SAME'
    h_conv5, w5, h5 = buildConv2D('Wconv3', 'bconv3', w4, h4, d4, max_pool4, filter_size5, n_filter5, stride5, pad5)
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
    h_fc_drop = buildFc('Wfc1', 'bfc1', h_conv5_flat,w5,h5,d5,n_fc,keep_prob)
    h_relu = tf.nn.relu(h_fc_drop)
    y_logits = buildSoftMax('Wsoft1','bsoft1',h_relu,n_fc,classes)

    return y_logits

if __name__ == '__main__':

    # prep for args
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetTrain', default='25dpf_60indiv_22000ImPerInd_rotateAndCrop', type = str)
    parser.add_argument('--datasetTest', default=None, type = str)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--ckpt_folder', default = "./ckpt_dir", type= str)
    parser.add_argument('--loadCkpt_folder', default = "", type = str)
    parser.add_argument('--num_indiv', default = 60, type = int)
    parser.add_argument('--num_train', default = 22000, type = int)
    parser.add_argument('--num_test', default = 0, type = int)
    parser.add_argument('--num_epochs', default = 500, type = int)
    parser.add_argument('--batch_size', default = 250, type = int)
    args = parser.parse_args()

    pathTrain = args.datasetTrain
    pathTest = args.datasetTest
    num_indiv = args.num_indiv
    num_train = args.num_train
    num_test = args.num_test
    ckpt_folder = args.ckpt_folder
    loadCkpt_folder = args.loadCkpt_folder

    # numIndiv, imsize, X_train, X_valid, X_test, Y_train, Y_valid, Y_test = dataHelper0(path, num_train, num_test, num_valid, ckpt_folder)
    numIndiv, imsize, X_train, Y_train, X_val, Y_val, X_test, Y_test = loadDataBase(pathTrain, num_indiv, num_train, num_test,ckpt_folder,pathTest)
    print 'val size:    images  labels'
    print X_val.shape, Y_val.shape
    resolution = np.prod(imsize)
    classes = numIndiv

    x = tf.placeholder(tf.float32, [None, resolution])
    y = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)

    y_logits = model(x,y,imsize[1],imsize[2],imsize[0],classes,keep_prob)


    # Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    # opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    opt = tf.train.GradientDescentOptimizer(0.01)
    optimizer = opt.minimize(cross_entropy)

    # Monitor accuracy
    prediction = tf.argmax(y_logits, 1)
    truth = tf.argmax(y,1)
    correct_prediction = tf.equal(prediction, truth)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # Create counter for epochs and savers
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver_model = tf.train.Saver([v for v in tf.all_variables() if "soft" not in v.name])
    saver_softmax = tf.train.Saver([v for v in tf.all_variables() if "soft" in v.name])

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        # Create folder for checkpoint if does not exist
        ckpt_dir = args.ckpt_folder
        ckpt_dir_model = ckpt_dir + "/model"
        ckpt_dir_softmax = ckpt_dir + "/softmax"
        if not os.path.exists(ckpt_dir): # Checkpoint folder does not exist
            os.makedirs(ckpt_dir) # we create a checkpoint folder
            os.makedirs(ckpt_dir_model)
            os.makedirs(ckpt_dir_softmax)
            print "Checkpoint folder created"
            'asdfasdf'
            if loadCkpt_folder: # we load weight of another model (knowledge transfer)
                ckpt_kt = tf.train.get_checkpoint_state(loadCkpt_folder + "/model")
                if ckpt_kt and ckpt_kt.model_checkpoint_path:
                    print ckpt_kt.model_checkpoint_path
                    saver_model.restore(sess, ckpt_kt.model_checkpoint_path) # restore all variables

        # load state of the training from the checkpoint
        ckpt_model = tf.train.get_checkpoint_state(ckpt_dir_model)
        ckpt_softmax = tf.train.get_checkpoint_state(ckpt_dir_softmax)
        if ckpt_model and ckpt_model.model_checkpoint_path:
            print ckpt_model.model_checkpoint_path
            saver_model.restore(sess, ckpt_model.model_checkpoint_path) # restore model variables
        if ckpt_softmax and ckpt_softmax.model_checkpoint_path:
            print ckpt_softmax.model_checkpoint_path
            saver_softmax.restore(sess, ckpt_softmax.model_checkpoint_path) # restore softmax

        # counter for epochs
        start = global_step.eval() # get last global_step
        print "Start from:", start

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
        if args.train == 1:
            if start == 0:
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
                for iter_i in range(iter_per_epoch-1):
                    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    loss, acc, pred, tr = sess.run([cross_entropy,accuracy, prediction, truth],
                                    feed_dict={
                                        x: batch_xs,
                                        y: batch_ys,
                                        keep_prob: 1.0
                                    })
                    lossEpoch.append(loss)
                    accEpoch.append(acc)

                    indivPred = [np.true_divide(np.sum(np.logical_and(np.equal(pred, i), np.equal(tr, i)), axis=0), np.sum(np.equal(tr, i))) for i in range(classes)]
                    indivAccEpoch.append(indivPred)

                    if iter_i % round(np.true_divide(iter_per_epoch,4)) == 0:
                        indivPred = np.around(indivPred, decimals=2)
                        print "Iter " + str(iter_i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc) #+ ", Individual Accuracy= "
                        # pprint(indivPred)

                    sess.run(optimizer, feed_dict={
                        x: batch_xs, y: batch_ys, keep_prob: 1})

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
                for Viter_i in range(val_iter_per_epoch-1):
                    Vbatch_xs = X_val[Vindices[Viter_i]:Vindices[Viter_i+1]]
                    Vbatch_ys = Y_val[Vindices[Viter_i]:Vindices[Viter_i+1]]

                    valLoss, valAcc, valPred, valTr = sess.run([cross_entropy, accuracy, prediction, truth],
                             feed_dict={
                                 x: Vbatch_xs,
                                 y: Vbatch_ys,
                                 keep_prob: 1.0
                             })
                    lossValEpoch.append(valLoss)
                    accValEpoch.append(valAcc)
                    valIndivPred = [np.true_divide(np.sum(np.logical_and(np.equal(valPred, i), np.equal(valTr, i)), axis=0), np.sum(np.equal(valTr, i))) for i in range(classes)]
                    valIIndivAccEpoch.append(valIndivPred)
                    if Viter_i % round(np.true_divide(val_iter_per_epoch,4)) == 0:
                        valIndivPred = np.around(valIndivPred, decimals=2)
                        print 'Validation Accuracy (batch = %d): ' % Viter_i + str(valAcc) #+ " Individual accuracy: "
                        # pprint(valIndivPred)

                meanValIndiviAcc = np.nanmean(valIIndivAccEpoch, axis=0)
                meanAcc = np.mean(meanValIndiviAcc)
                print 'Validation Accuracy (epoch = %d): ' % (start + epoch_i) + str(meanAcc) #+ "Individual Accuracy"
                pprint(meanValIndiviAcc)

                # print('Validation Mean accuracy (%d): ' % epoch_i + str(np.mean(accValEpoch)))

                lossPlot.append(np.mean(lossEpoch))
                accPlot.append(np.mean(accEpoch))
                valLossPlot.append(np.mean(lossValEpoch))
                valAccPlot.append(np.mean(accValEpoch))
                indivAccPlot.append(meanIndivAcc)
                indivValAccPlot.append(meanValIndiviAcc)

                lossAccDict = {
                    'loss': lossPlot,
                    'acc': accPlot,
                    'valLoss': valLossPlot,
                    'valAcc': valAccPlot,
                    'indivAcc': indivAccPlot,
                    'indivValAcc': indivValAccPlot,
                    }

                pickle.dump( lossAccDict , open( ckpt_dir_model + "/lossAcc.pkl", "wb" ) )

                # Poltter
                CNNplotterFast(lossPlot,accPlot,valAccPlot,valLossPlot,meanIndivAcc,meanValIndiviAcc)
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)
        if args.train == 0:
            # REMARK: the train on different animals has to be tested by considering the features, otherwise it would make no sense
            # one cannot associate different animals to the same labels of the training set!!!
            test_size = len(Y_test)
            test_iter = int(np.ceil(np.true_divide(test_size,batch_size)))
            print "Test size:", test_size
            print "Batch size:", batch_size
            print "Test Iterations", test_iter

            indices = np.linspace(0, test_size, test_iter)
            indices = indices.astype('int')
            testAcc = []
            for iter_i in range(test_iter-1):
                batch_xs = X_test[indices[iter_i]:indices[iter_i+1]]
                batch_ys = Y_test[indices[iter_i]:indices[iter_i+1]]
                acc = sess.run(accuracy,
                         feed_dict={
                             x: batch_xs,
                             y: batch_ys,
                             keep_prob: 1.0
                         })
                print('Batch accuracy for (%s): ' % ckpt_model.model_checkpoint_path + str(acc))
                testAcc.append(acc)
            print 'Average accuracy' + str(np.mean(testAcc))
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
