import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import h5py
import argparse




def dataHelper(path, num_train, num_test, num_valid):
    Fish = {}
    with h5py.File(path, 'r') as f:
        Fish['data'] = f['images']['data'][()]
        Fish['labels'] = f['images']['labels'][()]
        print 'database loaded'

    data = Fish['data'].astype(np.float32)
    # get image dimensions
    imsize = data[0].shape
    resolution = np.prod(imsize)
    # get labels
    label = Fish['labels'].astype(np.int32)
    label = np.squeeze(label)

    # compute number of individuals (classes)
    numIndiv = len(list(set(label)))
    # get labels in {0,...,numIndiv-1}
    label = np.subtract(label,1)

    print 'splitting data in train, test and validation'
    N = num_train*numIndiv # of training data
    N_test = num_test*numIndiv # of test data
    N_val = num_valid*numIndiv # validation data

    X_train = data[:N]
    X_test = data[N:N+N_test]
    X_valid = data[N+N_test:N+N_test+N_val]
    y_train = label[:N]
    y_test = label[N:N+N_test]
    y_valid = label[N+N_test:N+N_test+N_val]

    # reshape images
    X_train = np.reshape(X_train, [N, resolution])
    X_test = np.reshape(X_test, [N_test, resolution])
    X_valid = np.reshape(X_valid, [N_val, resolution])
    # dense to one hot, i.e. [i]-->[0,0,...0,1 (ith position),0,..,0]
    Y_train = dense_to_one_hot(y_train, n_classes=numIndiv)
    Y_valid = dense_to_one_hot(y_valid, n_classes=numIndiv)
    Y_test = dense_to_one_hot(y_test, n_classes=numIndiv)

    return numIndiv, imsize, X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def spTrans(x_tensor,width, height, channels, n_loc,keep_prob):
    resolution = width * height * channels
    W_fc_loc1 = weight_variable([resolution, n_loc])
    b_fc_loc1 = bias_variable([n_loc])

    W_fc_loc2 = weight_variable([n_loc, 6])
    # Use identity transformation as starting point
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    # Two layer localisation network
    h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
    # dropout (reduce overfittin)
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
    # %% Second layer
    h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
    # spatial transformer
    out_size = (width, height)
    h_trans = transformer(x_tensor, h_fc_loc2, out_size)

    return h_trans, b_fc_loc2, h_fc_loc2

def computeVolume(width, height, strides):
    c1 = float(strides[1])
    c2 = float(strides[2])
    widthS = int(np.ceil(width/c1))
    heightS = int(np.ceil(height/c2))
    return widthS, heightS

def buildConv2D(inputWidth, inputHeight, inputConv, n_input_filters,filter_size, n_filters, stride, pad):
    WConv = weight_variable([filter_size, filter_size, n_input_filters, n_filters])
    bConv = bias_variable([n_filters])
    hConv = tf.nn.relu(
        tf.nn.conv2d(input=inputConv,
                     filter=WConv,
                     strides=stride,
                     padding=pad) +
        bConv)
    w,h = computeVolume(inputWidth, inputHeight, stride)
    return hConv,w,h

def buildFc(inputFc,height,width,n_filters,n_fc,keep_prob):
    W_fc = weight_variable([height * width * n_filters, n_fc])
    b_fc = bias_variable([n_fc])
    h_fc = tf.nn.relu(tf.matmul(inputFc, W_fc) + b_fc)
    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)
    return h_fc_drop

def buildSoftMax(inputSoftMax,n_fc,classes):
    W_fc = weight_variable([n_fc, classes])
    b_fc = bias_variable([classes])
    y_logits = tf.matmul(inputSoftMax, W_fc) + b_fc
    return y_logits



def model(x,y,width, height, channels, classes, n_loc,keep_prob):
    x_tensor = tf.reshape(x, [-1, width, height, channels])
    # spatial transformer
    h_trans, b_fc_loc2, h_fc_loc2 = spTrans(x_tensor,width, height, channels, n_loc,keep_prob)
    # conv1
    filter_size1 = 3
    n_filter1 = 16
    stride1 = [1,2,2,1]
    pad1 = "SAME"
    h_conv1, w1, h1 = buildConv2D(width, height, h_trans, 1, filter_size1, n_filter1, stride1, pad1)
    # conv2
    filter_size2 = 3
    n_filter2 = 64
    stride2 = [1,2,2,1]
    pad2 = "SAME"
    h_conv2, w2, h2 = buildConv2D(w1, h1, h_conv1, n_filter1, filter_size1, n_filter2, stride2, pad2)
    # conv3
    filter_size3 = 3
    n_filter3 = 128
    stride3 = [1,2,2,1]
    pad3 = "SAME"
    h_conv3, w3, h3 = buildConv2D(w2, h2, h_conv2, n_filter2, filter_size1, n_filter3, stride3, pad3)
    # linearize weights for fully-connected layer
    resolutionS = w3 * h3
    h_conv3_flat = tf.reshape(h_conv3, [-1, resolutionS*n_filter3])
    # fully-connected 1
    n_fc = 2048
    h_fc_drop = buildFc(h_conv3_flat,w3,h3,n_filter3,n_fc,keep_prob)
    y_logits = buildSoftMax(h_fc_drop,n_fc,classes)

    return y_logits, b_fc_loc2, h_fc_loc2, h_trans

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

    y_logits, b_fc_loc2,h_fc_loc2, h_trans = model(x,y,imsize[1],imsize[2],imsize[0],classes,20,keep_prob)


    # %% Define loss/eval/training functions
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
    opt = tf.train.AdamOptimizer()
    optimizer = opt.minimize(cross_entropy)
    grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

    # %% Monitor accuracy
    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    ckpt_dir = args.ckpt_folder
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver()
    # train = 1
    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

        start = global_step.eval() # get last global_step
        # print "Start from:", start

        # %% We'll now train in minibatches and report accuracy, loss:
        iter_per_epoch = args.itsPerEpoch
        n_epochs = args.num_epochs
        train_size = args.num_train

        indices = np.linspace(0, train_size - 1, iter_per_epoch)
        indices = indices.astype('int')
        if args.train == 1:
            print "Start from:", start
            for epoch_i in range(start, n_epochs):
                for iter_i in range(iter_per_epoch - 1):
                    batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
                    batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

                    if iter_i % 10 == 0:
                        loss = sess.run(cross_entropy,
                                        feed_dict={
                                            x: batch_xs,
                                            y: batch_ys,
                                            keep_prob: 1.0
                                        })
                        print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

                    sess.run(optimizer, feed_dict={
                        x: batch_xs, y: batch_ys, keep_prob: 0.8})
                # convolve = sess.run(h_conv3, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print(convolve[0].shape)

                global_step.assign(epoch_i).eval() # set and update(eval) global_step with index, i
                saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

                print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                                                                 feed_dict={
                                                                     x: X_valid,
                                                                     y: Y_valid,
                                                                     keep_prob: 1.0
                                                                 })))
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
            testInd = 2
            theta = sess.run(h_trans, feed_dict={
                   x: X_test, keep_prob: 1.0})
            # print(theta[0])
            testImage = np.reshape(X_test[testInd], [imsize[1], imsize[2]])
            plt.subplot(121)
            plt.imshow(testImage)
            # transf = tf.reshape(theta[0], [2,3])
            # print theta[0].shape
            plt.subplot(122)
            plt.imshow(np.squeeze(theta[testInd]))
            plt.show()
