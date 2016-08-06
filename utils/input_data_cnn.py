import numpy as np
import h5py
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import pickle
import os

def dataHelper(path, num_train, num_test, num_valid, ckpt_folder):
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

    print 'permuting the dataset'
    if not os.path.exists(ckpt_folder):
        perm = np.random.permutation(len(label))
        # Save a permutation into a pickle file.
        permutation = { "perm": perm }
        pickle.dump( permutation, open( "../temp/permutation.p", "wb" ) )
        print ' No permutation exists, new one created'
    else:
        permutation = pickle.load( open( "../temp/permutation.p", "rb" ) )
        print ' Permutation loaded'
        perm = permutation['perm']

    label = label[perm]
    data = data[perm]


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
