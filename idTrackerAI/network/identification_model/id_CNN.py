from __future__ import absolute_import, division, print_function
import os
import sys

import tensorflow as tf
import numpy as np
import logging

from cnn_architectures import cnn_model_0, \
                                cnn_model_1, \
                                cnn_model_2, \
                                cnn_model_3, \
                                cnn_model_4, \
                                cnn_model_5, \
                                cnn_model_6, \
                                cnn_model_7, \
                                cnn_model_8, \
                                cnn_model_9, \
                                cnn_model_10, \
                                cnn_model_11

CNN_MODELS_DICT = {0: cnn_model_0,
                    1: cnn_model_1,
                    2: cnn_model_2,
                    3: cnn_model_3,
                    4: cnn_model_4,
                    5: cnn_model_5,
                    6: cnn_model_6,
                    7: cnn_model_7,
                    8: cnn_model_8,
                    9: cnn_model_9,
                    10: cnn_model_10,
                    11: cnn_model_11}
logger = logging.getLogger("__main__.id_CNN")

class ConvNetwork():
    def __init__(self, params, training_flag = True, restore_index = None):
        """CNN
        params (NetworkParams object)
        training_flag (bool)
            True to backpropagate
        restore_index (integer)
            checkpoint to be used in restoring the network
        """
        # Set main attibutes of the class
        self.image_width = params.image_size[0]
        self.image_height = params.image_size[1]
        self.image_channels = params.image_size[2]
        self.params = params
        self.restore_index = restore_index
        # Initialize layers to optimize to be empty. This means tha we will
        # optimize all the layers of our network
        self.layers_to_optimise = None
        self.training = training_flag
        # Build graph with the network, loss, optimizer and accuracies
        tf.reset_default_graph()
        self._build_graph()
        self.set_savers()
        # Create list of operations to run during training and validation
        if self.training:
            self.ops_list = [self.loss, self.accuracy, self.individual_accuracy]
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            logger.debug('Restoring...')
            # Get subfolders from where we will load the network from previous checkpoints
            [self.restore_folder_conv,self.restore_folder_fc_softmax] = get_checkpoint_subfolders( self.params._restore_folder, ['conv', 'softmax'])
        elif self.is_knowledge_transfer:
            # Get subfolders from where we will load the convolutional filters to perform knowledge transfer
            logger.debug('Performing knowledge transfer...')
            [self.restore_folder_conv] = get_checkpoint_subfolders(self.params._knowledge_transfer_folder,['conv'])
            self.session.run(self.global_step.assign(0))
        # self.restore()
        if self.training:
            self.create_summaries_writers()

    @property
    def is_knowledge_transfer(self):
        if self.params._knowledge_transfer_folder is not None:
            self.restore_folder_fc_softmax = None
            logger.debug("restore_folder_fc_softmax:", self.restore_folder_fc_softmax)
        return self.params._knowledge_transfer_folder is not None

    @property
    def is_restoring(self):
        return self.params._restore_folder is not None

    # Create savers for the convolutions and the fully conected and softmax separately
    def set_savers(self):
        self.saver_conv = createSaver('saver_conv', exclude_fc_and_softmax = True)
        self.saver_fc_softmax = createSaver('saver_fc_softmax', exclude_fc_and_softmax = False)
        # Create subfolders where we will save the checkpoints of the trainig
        [self.save_folder_conv,self.save_folder_fc_softmax] = create_checkpoint_subfolders( self.params._save_folder, ['conv', 'softmax'])

    def create_summaries_writers(self):
        self.summary_op = tf.summary.merge_all()
        self.summary_writer_training = tf.summary.FileWriter(self.params._save_folder + '/train',self.session.graph)
        self.summary_writer_validation = tf.summary.FileWriter(self.params._save_folder + '/val',self.session.graph)

    def _build_graph(self):
        self.x_pl = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_channels], name = 'images')
        # self.x_pl = tf.placeholder(tf.float32, [None, None, None, self.image_channels], name = 'images')
        # self.y_logits, self.conv_vector = cnn_model(self.x_pl,self.params.number_of_animals)
        # self.y_logits, self.fc_vector, (self.W1, self.W2, self.W3, self.WFC, self.WSoft) = cnn_model_0(self.x_pl,self.params.number_of_animals)
        # self.y_logits = cnn_model_0(self.x_pl,self.params.number_of_animals)
        logger.debug('training model %i' %self.params.cnn_model)
        self.y_logits = CNN_MODELS_DICT[self.params.cnn_model](self.x_pl,self.params.number_of_animals, self.image_width, self.image_height, self.image_channels)

        self.softmax_probs = tf.nn.softmax(self.y_logits)
        self.predictions = tf.cast(tf.add(tf.argmax(self.softmax_probs,1),1),tf.float32)

        if self.training:
            self.y_target_pl = tf.placeholder(tf.float32, [None, self.params.number_of_animals], name = 'labels')
            self.keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')
            self.loss_weights_pl = tf.placeholder(tf.float32, [None], name = 'loss_weights')
            self.loss = self.weighted_loss()
            self.optimisation_step, self.global_step = self.set_optimizer()
            # self.accuracy = tf.accuracy()
            self.accuracy, self.individual_accuracy = self.evaluation()

    def get_layers_to_optimize(self):
        if self.params.scopes_layers_to_optimize is not None:
            self.layers_to_optimise = []
            for scope_layer in self.params.scopes_layers_to_optimize:
                with tf.variable_scope(scope_layer, reuse=True) as scope:
                    W = tf.get_variable("weights")
                    B = tf.get_variable("biases")
                self.layers_to_optimise.append([W, B])
        self.layers_to_optimise = [layer for layers in self.layers_to_optimise for layer in layers]

    def weighted_loss(self):
        cross_entropy = tf.reduce_mean(
            tf.contrib.losses.softmax_cross_entropy(self.y_logits,self.y_target_pl, self.loss_weights_pl), name = 'CrossEntropyMean')
        # cross_entropy = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=y_logits,labels=y), name = 'CrossEntropyMean')
        self._add_loss_summary(cross_entropy)
        return cross_entropy

    def compute_loss_weights(self, training_labels):
        self.weights = 1. - np.sum(training_labels,axis=0) / len(training_labels)

    def set_optimizer(self):
        if not self.params.use_adam_optimiser:
            logger.debug('Training with SGD')
            optimizer = tf.train.GradientDescentOptimizer(self.params.learning_rate)
        elif self.params.use_adam_optimiser:
            logger.debug('Training with ADAM')
            optimizer = tf.train.AdamOptimizer(learning_rate = self.params.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.params.scopes_layers_to_optimize is not None:
            logger.debug('Optimizing %s' %self.params.scopes_layers_to_optimize)
            self.get_layers_to_optimize()
            train_op = optimizer.minimize(self.loss, var_list = self.layers_to_optimise)
        else:
            logger.debug('Optimizing the whole network')
            train_op = optimizer.minimize(loss=self.loss)
        return train_op, global_step

    def evaluation(self):
        individual_accuracy = compute_individual_accuracy(self.y_target_pl, self.y_logits, self.params.number_of_animals)
        accuracy = compute_accuracy(self.y_target_pl, self.y_logits)
        return accuracy, individual_accuracy

    def reinitialize_softmax_and_fully_connected(self):
        logger.debug('Reinitializing softmax and fully connected')
        self.session.run(tf.variables_initializer([v for v in tf.global_variables() if 'soft' in v.name or 'full' in v.name]))

    def restore(self):
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            ckpt = tf.train.get_checkpoint_state(self.restore_folder_conv)
            if self.restore_index is None:
                self.saver_conv.restore(self.session, ckpt.model_checkpoint_path) # restore convolutional variables
                logger.debug('Restoring convolutional part from %s' %ckpt.model_checkpoint_path)
            else:
                self.saver_conv.restore(self.session, ckpt.all_model_checkpoint_paths[self.restore_index])
                logger.debug('Restoring convolutional part from %s' %ckpt.all_model_checkpoint_paths[self.restore_index])

            ckpt = tf.train.get_checkpoint_state(self.restore_folder_fc_softmax)
            if self.restore_index is None:
                self.saver_fc_softmax.restore(self.session, ckpt.model_checkpoint_path) # restore fully-conected and softmax variables
                logger.debug('Restoring fully-connected and softmax part from %s' %ckpt.model_checkpoint_path)
            else:
                self.saver_fc_softmax.restore(self.session, ckpt.all_model_checkpoint_paths[self.restore_index]) # restore fully-conected and softmax variables
                logger.debug('Restoring fully-connected and softmax part from %s' %ckpt.all_model_checkpoint_paths[self.restore_index])


    def compute_batch_weights(self, batch_labels):
        # if self.weighted_flag:
        batch_weights = np.sum(self.weights*batch_labels,axis=1)
        # elif not self.weighted_flag:
        #     batch_weights = np.ones(len(labels_feed))

        return batch_weights

    def get_feed_dict(self, batch):
        (batch_images, batch_labels) = batch
        batch_weights = self.compute_batch_weights(batch_labels)
        return  { self.x_pl: batch_images,
                  self.y_target_pl: batch_labels,
                  self.loss_weights_pl: batch_weights,
                  self.keep_prob_pl: self.params.keep_prob}

    def train(self,batch):
        feed_dict = self.get_feed_dict(batch)
        self.session.run(self.optimisation_step,feed_dict = feed_dict)
        outList = self.session.run(self.ops_list, feed_dict = feed_dict)
        return outList, feed_dict

    def validate(self,batch):
        feed_dict = self.get_feed_dict(batch)
        outList = self.session.run(self.ops_list, feed_dict = feed_dict)
        return outList, feed_dict

    def predict(self,batch):
        feed_dict = {self.x_pl: batch}
        return self.session.run([self.softmax_probs,self.predictions], feed_dict = feed_dict)

    def get_fully_connected_vectors(self,batch):
        feed_dict = {self.x_pl: batch}
        return self.session.run(self.fc_vector, feed_dict = feed_dict)

    def write_summaries(self,epoch_i,feed_dict_train, feed_dict_val):
        summary_str_training = self.session.run(self.summary_op, feed_dict=feed_dict_train)
        summary_str_validation = self.session.run(self.summary_op, feed_dict=feed_dict_val)
        self.summary_writer_training.add_summary(summary_str_training, epoch_i)
        self.summary_writer_validation.add_summary(summary_str_validation, epoch_i)

    def _add_loss_summary(self,loss):
        tf.summary.scalar(loss.op.name, loss)

    def save(self):
        self.saver_conv.save(self.session, os.path.join(self.save_folder_conv, "conv.ckpt"), global_step = self.global_step)
        self.saver_fc_softmax.save(self.session, os.path.join(self.save_folder_fc_softmax, "softmax.ckpt"), global_step = self.global_step)

def compute_individual_accuracy(labels,logits,classes):
    # We add 1 to the labels and predictions to avoid having a 0 label
    labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
    predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
    labelsRep = tf.reshape(tf.tile(labels, [classes]), [classes,tf.shape(labels)[0]])
    correct = tf.cast(tf.equal(labels,predictions),tf.float32)
    indivCorrect = tf.multiply(predictions,correct)
    indivRep = tf.cast(tf.transpose(tf.reshape(tf.tile(tf.range(1,classes+1), [tf.shape(labels)[0]]), [tf.shape(labels)[0],classes])),tf.float32)
    indivCorrectRep = tf.reshape(tf.tile(indivCorrect, [classes]), [classes,tf.shape(labels)[0]])
    correctPerIndiv = tf.cast(tf.equal(indivRep,indivCorrectRep),tf.float32)
    countCorrect = tf.reduce_sum(correctPerIndiv,1)
    numImagesPerIndiv = tf.reduce_sum(tf.cast(tf.equal(labelsRep,indivRep),tf.float32),1)
    indivAcc = tf.div(countCorrect,numImagesPerIndiv)
    return indivAcc

def compute_accuracy(labels, logits):
    # We add 1 to the labels and predictions to avoid having a 0 label
    labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
    predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
    # acc = tf.metrics.accuracy(labels, predictions)
    correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
    acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
    return acc


def get_checkpoint_subfolders(folderName, subfoldersNameList):
    '''
    create if it does not exist the folder folderName in CNN and
    the same for the subfolders in the subfoldersNameList
    '''
    subPaths = []
    for name in subfoldersNameList:
        subPath = folderName + '/' + name
        if not os.path.exists(subPath):
            os.makedirs(subPath)
            logger.debug('%s has been created' %subPath)
        else:
            logger.debug('%s already exists' %subPath)
        subPaths.append(subPath)
    return subPaths

def create_checkpoint_subfolders(folderName, subfoldersNameList):
    '''
    create if it does not exist the folder folderName in CNN and
    the same for the subfolders in the subfoldersNameList
    '''
    subPaths = []
    for name in subfoldersNameList:
        subPath = folderName + '/' + name
        if not os.path.exists(subPath):
            os.makedirs(subPath)
            print(subPath + ' has been created')
        else:
            print(subPath + ' already exists')
        subPaths.append(subPath)
    return subPaths

def createSaver(name, exclude_fc_and_softmax):
    if not exclude_fc_and_softmax:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' in v.name or 'full' in v.name], name = name, max_to_keep = 1000000)
    elif exclude_fc_and_softmax:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'conv' in v.name], name = name, max_to_keep = 1000000)
    else:
        raise ValueError('The second argument has to be a boolean')

    return saver
