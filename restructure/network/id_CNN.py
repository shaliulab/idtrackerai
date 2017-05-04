from __future__ import absolute_import, division, print_function
import os

import tensorflow as tf
import numpy as np

from cnn_architectures import cnn_model

IMAGE_SIZE = (32,32,1)

class ConvNetwork():
    def __init__(self, params):
        # Set main attibutes of the class
        self.image_width = IMAGE_SIZE[0]
        self.image_height = IMAGE_SIZE[1]
        self.image_channels = IMAGE_SIZE[2]
        self.params = params
        # Initialize layers to optimize to be empty. This means tha we will
        # optimize all the layers of our network
        self.layers_to_optimise = []
        # Build graph with the network, loss, optimizer and accuracies
        self._build_graph()
        # Create savers for the convolutions and the fully conected and softmax separately
        self.saver_conv = createSaver('saver_conv', exclude_fc_and_softmax = True)
        self.saver_fc_softmax = createSaver('saver_fc_softmax', exclude_fc_and_softmax = False)
        # Create list of operations to run during training and validation
        self.ops_list = [self.loss, self.accuracy, self.individual_accuracy]
        # Create subfolders where we will save the checkpoints of the trainig
        [self.save_folder_conv,self.save_folder_fc_softmax] = create_checkpoint_subfolders( self.params._save_folder, ['conv', 'softmax'])

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            # Get subfolders from where we will load the network from previous checkpoints
            [self.restore_folder_conv,self.restore_folder_fc_softmax] = get_checkpoint_subfolders( self.params._restore_folder, ['conv', 'softmax'])
        elif self.is_knowledge_transfer:
            # Get subfolders from where we will load the convolutional filters to perform knowledge transfer
            print('\nPerforming knowledge transfer...')
            [self.restore_folder_conv] = get_checkpoint_subfolders(self.params._knowledge_transfer_folder,['conv'])
            self.session.run(self.global_step.assign(0))
        self.restore()

        self.summary_op = tf.summary.merge_all()
        self.summary_writer_training = tf.summary.FileWriter(self.params._save_folder + '/train',self.session.graph)
        self.summary_writer_validation = tf.summary.FileWriter(self.params._save_folder + '/val',self.session.graph)

    @property
    def is_knowledge_transfer(self):
        if self.params._knowledge_transfer_folder is not None:
            self.restore_folder_fc_softmax = None
        return self.params._knowledge_transfer_folder is not None

    @property
    def is_restoring(self):
        return self.params._restore_folder is not None

    def _build_graph(self):
        self.x_pl = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_channels], name = 'images')
        self.y_target_pl = tf.placeholder(tf.float32, [None, self.params.number_of_animals], name = 'labels')
        self.keep_prob_pl = tf.placeholder(tf.float32, name = 'keep_prob')
        self.loss_weights_pl = tf.placeholder(tf.float32, [None], name = 'loss_weights')

        self.y_logits = cnn_model(self.x_pl,self.params.number_of_animals)

        self.loss = self.weighted_loss(self.y_target_pl, self.y_logits, self.loss_weights_pl)

        self.optimisation_step, self.global_step = self.set_optimizer(self.loss)
        self.accuracy, self.individual_accuracy = self.evaluation(self.y_target_pl, self.y_logits)

        self.softmax_probs = tf.nn.softmax(self.y_logits)
        self.predictions = tf.cast(tf.add(tf.argmax(self.softmax_probs,1),1),tf.float32)

    def get_layers_to_optimize(self):
        if self.params.scopes_layers_to_optimize is not None:
            for scope_layer in self.params.scopes_layers_to_optimize:
                with tf.variable_scope(scope_layer, reuse=True) as scope:
                    W = tf.get_variable("weights")
                    B = tf.get_variable("biases")
                self.layers_to_optimise.append([W, B])
        self.layers_to_optimise = [layer for layers in self.layers_to_optimise for layer in layers]

    def weighted_loss(self,y, y_logits, loss_weights):
        cross_entropy = tf.reduce_mean(
            tf.contrib.losses.softmax_cross_entropy(y_logits,y, loss_weights), name = 'CrossEntropyMean')
        # cross_entropy = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=y_logits,labels=y), name = 'CrossEntropyMean')
        self._add_loss_summary(cross_entropy)
        return cross_entropy

    def compute_loss_weights(self, training_labels):
        self.weights = 1. - np.sum(training_labels,axis=0) / len(training_labels)

    def set_optimizer(self, loss):
        if not self.params.use_adam_optimiser:
            print('\nTraining with SGD')
            optimizer = tf.train.GradientDescentOptimizer(self.params.learning_rate)
        elif self.params.use_adam_optimiser:
            print('\nTraining with ADAM')
            optimizer = tf.train.AdamOptimizer(learning_rate = self.params.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.params.scopes_layers_to_optimize is not None:
            print('\nOptimizing ', self.params.scopes_layers_to_optimize)
            self.get_layers_to_optimize()
            train_op = optimizer.minimize(loss, var_list = self.layers_to_optimise)
        else:
            print('\nOptimizing the whole network')
            train_op = optimizer.minimize(loss=loss)
        return train_op, global_step

    def evaluation(self,y,y_logits):
        individual_accuracy = compute_individual_accuracy(y,y_logits, self.params.number_of_animals)
        accuracy = compute_accuracy(y,y_logits)
        return accuracy, individual_accuracy

    def restore(self):
        self.session.run(tf.global_variables_initializer())
        try:
            ckpt = tf.train.get_checkpoint_state(self.restore_folder_conv)
            self.saver_conv.restore(self.session, ckpt.model_checkpoint_path) # restore convolutional variables
            print('\nRestoring convolutional part from ', ckpt.model_checkpoint_path)
        except:
            print('\nWarning: no checkpoints found for the convolutional part')
        if self.is_restoring:
            try:
                ckpt = tf.train.get_checkpoint_state(self.restore_folder_fc_softmax)
                self.saver_fc_softmax.restore(self.session, ckpt.model_checkpoint_path) # restore fully-conected and softmax variables
                print('\nRestoring fully-connected and softmax part from ', ckpt.model_checkpoint_path)
            except:
                print('\nWarning: no checkpoints found for the fully-connected and softmax parts')

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
        outList = self.session.run([self.softmax_probs,self.predictions], feed_dict = feed_dict)
        return outList

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
            print(subPath + ' has been created')
        else:
            print(subPath + ' already exists')
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
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' in v.name or 'full' in v.name], name = name)
    elif exclude_fc_and_softmax:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' not in v.name or 'full' not in v.name], name = name)
    else:
        raise ValueError('The second argument has to be a boolean')

    return saver
