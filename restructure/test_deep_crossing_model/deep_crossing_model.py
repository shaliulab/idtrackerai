import tensorflow as tf
import os
import sys
sys.path.append('../utils')
# from cnn_utils import *

import numpy as np

class ConvNetwork():
    def __init__(self, from_video_path = None, weight_positive = 1, architecture = None, learning_rate = 0.01):
        self.sesh = tf.Session()
        self.weight_positive = weight_positive
        self.architecture = architecture
        self.learning_rate = learning_rate
        handles = self._build_graph()
        print("Building graph....")
        (self.X, self.Y, self.Y_target, self.loss, self.accuracy, self.train_step) = handles
        self.saver = tf.train.Saver()

        if from_video_path == None:
            self.sesh.run(tf.global_variables_initializer())
        else:
            self.restore(from_video_path)

    def _build_graph(self):
        images = tf.placeholder(tf.float32, shape = [None, 294, 294, 1])
        Y_logits = self.architecture(images, 2, 294, 294, 1)
        Y_target = tf.placeholder(tf.float32, shape = [None, 2])
        loss = self.weighted_loss(Y_logits, Y_target, self.weight_positive)
        # train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        accuracy = self.compute_accuracy(Y_target, Y_logits)
        return (images, Y_logits, Y_target, loss, accuracy, train_step)

    @staticmethod
    def weighted_loss(Y_logits, Y_target, weight):
        cross_entropy = tf.reduce_mean(
            tf.contrib.losses.softmax_cross_entropy(Y_logits, Y_target, weight), name = 'CrossEntropyMean')
        return cross_entropy

    @staticmethod
    def compute_accuracy(labels, logits):
        # We add 1 to the labels and predictions to avoid having a 0 label
        labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
        predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
        # acc = tf.metrics.accuracy(labels, predictions)
        correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
        acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
        return acc

    def train(self,batch):
        (batch_images, batch_labels) = batch
        loss_value, acc_value, _ = self.sesh.run([self.loss, self.accuracy, self.train_step],
                                                    feed_dict={self.X: batch_images, self.Y_target: batch_labels})
        loss_mean = np.mean(loss_value)
        acc_mean = np.mean(acc_value)
        print("(training) loss: %f, acc: %f " %(loss_mean, acc_mean))

    def validate(self,batch):
        (batch_images, batch_labels) = batch
        loss_value, acc_value = self.sesh.run([self.loss, self.accuracy],
                                                    feed_dict={self.X: batch_images, self.Y_target: batch_labels})
        loss_mean = np.mean(loss_value)
        acc_mean = np.mean(acc_value)
        print("(validation) loss: %f, acc: %f " %(loss_mean, acc_mean))

    def prediction(self,images):
        return self.sesh.run(self.Y, feed_dict={self.X: images})

    def save(self, filename, global_step):
        save_path = self.saver.save(self.sesh, filename, global_step=global_step)
        print("Model saved in file:", save_path)

    def restore(self, from_video_path):
        video_folder = os.path.dirname(from_video_path)
        self.ckpt_path = os.path.join(video_folder, 'deep_crossings_ckpt')
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring from " + ckpt.model_checkpoint_path)
            self.saver.restore(self.sesh, ckpt.model_checkpoint_path)
        else:
            print("check if any checkpoint has been stored in " + self.ckpt_path + ". The folder seems empty, or non existent")
