import tensorflow as tf
import numpy as np
import sys
sys.path.append('../utils')
from py_utils import *
from siamese_utils import *

""" Constrastive Loss test (siamese_utils) """
def testConstrastive():
    # inputs
    features1 = [[1.,2.,3.], [4.,5.,6.]]
    features2 = [[1.,1.,1.], [2.,2.,2.]]
    targets = [1.,0.]
    # intermediate checks
    # LG = [1.,9.]
    # LI = [ 22.68836594, 9.01165676]

    # outputs
    loss = contrastive_loss1(features1, features2, targets, name=None)
    correctLoss = [ 1., 9.01165676]
    sess = tf.Session()
    loss = sess.run(loss)
    diff = np.subtract(loss,correctLoss)
    if diff.all() <0.000001:
        return "contrastive_loss1: test ok"
    else:
        return "contrastive_loss1: to be checked"
print testConstrastive()



# print Ncycle(range(5),0)
# Output: [2, 3, 4, 0, 1]

""" counRate test (py_utils) """
def testCountRateSet():
    counter = countRate([1.0,1.0,5.0,1.0,5.0,5.0,3.0])
    if counter  == [(1.0, 3), (5.0, 3), (3.0, 1)]:
        return "countRate: test ok"
    else:
        return "please check the function countRate"
print testCountRate()

""" deleteDuplicates test (py_utils) """
def testDeleteDuplicates():
    test = [[1,2,3],[1,2,3],[4,5,6]]
    newt, inds = deleteDuplicates(test)
    if ans == [[1,2,3],[4,5,6]] and inds == [1]:
        return "deleteDuplicates: test ok"
    else:
        return "please check the function deleteDuplicates"
