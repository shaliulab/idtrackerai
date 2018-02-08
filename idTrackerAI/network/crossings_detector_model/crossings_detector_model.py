from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import numpy as np
import sys
if sys.argv[0] == 'idtrackerdeepApp.py':
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.crossing_detector_model")


class ConvNetwork_crossings(object):
    def __init__(self, network_params):

        self.params = network_params
        self.architecture = network_params.architecture
        self.learning_rate = network_params.learning_rate
        self.image_size = network_params.image_size
        self.number_of_classes = network_params.number_of_classes
        logger.info("Building graph....")
        self._build_graph()
        self.ops_list = [self.loss, self.accuracy, self.individual_accuracy]
        self.saver = tf.train.Saver()
        self.sesh = tf.Session()
        self.sesh.run(tf.global_variables_initializer())

    def _build_graph(self):
        self.X = tf.placeholder(tf.float32, shape = [None, self.image_size[0], self.image_size[1], self.image_size[2]])
        self.Y_logits = self.architecture(self.X, self.number_of_classes, self.image_size[0], self.image_size[1], self.image_size[2])
        self.Y_target = tf.placeholder(tf.float32, shape = [None, 2])
        self.loss_weights_pl = tf.placeholder(tf.float32, [None], name = 'loss_weights')
        self.loss = self.weighted_loss()
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        self.accuracy = self.compute_accuracy(self.Y_target, self.Y_logits)
        self.individual_accuracy = self.compute_individual_accuracy(self.Y_target, self.Y_logits, self.number_of_classes)
        self.softmax_probs = tf.nn.softmax(self.Y_logits)
        self.predictions = tf.cast(tf.argmax(self.softmax_probs,1),tf.float32)

    def weighted_loss(self):
        cross_entropy = tf.reduce_mean(
            tf.contrib.losses.softmax_cross_entropy(self.Y_logits, self.Y_target, self.loss_weights_pl), name = 'CrossEntropyMean')
        return cross_entropy

    def compute_loss_weights(self, training_labels):
        self.weights = 1. - np.sum(training_labels, axis=0) / len(training_labels)

    @staticmethod
    def compute_accuracy(labels, logits):
        # We add 1 to the labels and predictions to avoid having a 0 label
        labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
        predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
        # acc = tf.metrics.accuracy(labels, predictions)
        correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
        acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
        return acc

    @staticmethod
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

    def compute_batch_weights(self, batch_labels):
        batch_weights = np.sum(self.weights*batch_labels,axis=1)
        return batch_weights

    def train(self,batch):
        (batch_images, batch_labels) = batch
        batch_weights = self.compute_batch_weights(batch_labels)
        feed_dict={self.X: batch_images, self.Y_target: batch_labels, self.loss_weights_pl: batch_weights}
        self.sesh.run(self.train_step, feed_dict = feed_dict)
        out_list = self.sesh.run(self.ops_list, feed_dict = feed_dict)
        return out_list, feed_dict

    def validate(self,batch):
        (batch_images, batch_labels) = batch
        batch_weights = self.compute_batch_weights(batch_labels)
        feed_dict={self.X: batch_images, self.Y_target: batch_labels, self.loss_weights_pl: batch_weights}
        self.sesh.run(self.train_step, feed_dict = feed_dict)
        out_list = self.sesh.run(self.ops_list, feed_dict = feed_dict)
        return out_list, feed_dict

    def prediction(self,batch_images):
        return self.sesh.run(self.predictions, feed_dict={self.X: batch_images})

    @property
    def is_knowledge_transfer(self):
        return self.params._knowledge_transfer_folder is not None

    @property
    def is_restoring(self):
        return self.params._restore_folder is not None

    def restore(self):
        self.sesh.run(tf.global_variables_initializer())
        try:
            ckpt = tf.train.get_checkpoint_state(self.params._restore_folder)
            logger.info("restoring crossings detector model from %s" %ckpt.model_checkpoint_path)
            self.saver.restore(self.sesh, ckpt.model_checkpoint_path) # restore convolutional variables
        except:
            logger.info('\nWarning: no checkpoints found in the folder %s' %self.params._restore_folder)

    def save(self, global_step):
        logger.info("(in save) self.params._save_folder %s" %self.params._save_folder)
        save_path = self.saver.save(self.sesh, os.path.join(self.params._save_folder,'.ckpt'), global_step=global_step)
        logger.info("Model saved in file: %s" %save_path)
