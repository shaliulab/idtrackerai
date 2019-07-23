# This file is part of idtracker.ai a multiple animals tracking system
# described in [1].
# Copyright (C) 2017- Francisco Romero Ferrero, Mattia G. Bergomi,
# Francisco J.H. Heras, Robert Hinz, Gonzalo G. de Polavieja and the
# Champalimaud Foundation.
#
# idtracker.ai is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (idtrackerai@gmail.com) or
# use the tools available at https://gitlab.com/polavieja_lab/idtrackerai.git.
#
# [1] Romero-Ferrero, F., Bergomi, M.G., Hinz, R.C., Heras, F.J.H., de Polavieja, G.G., Nature Methods, 2019.
# idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
# (F.R.-F. and M.G.B. contributed equally to this work.
# Correspondence should be addressed to G.G.d.P: gonzalo.polavieja@neuro.fchampalimaud.org)

import os
import sys

import tensorflow as tf
import numpy as np

from idtrackerai.network.cnn_architectures import cnn_model_0, \
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

"""Dictionary of the convolutional models that can be used
"""
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


if sys.argv[0] == 'idtrackeraiApp.py' or 'idtrackeraiGUI' in sys.argv[0]:
    from kivy.logger import Logger
    logger = Logger
else:
    import logging
    logger = logging.getLogger("__main__.id_CNN")


class ConvNetwork():
    """Manages the main Tensorflow graph for the convolutional network that
    identifies images (idCNN)

    Attributes
    ----------

    image_width : int
        Width of the input image to the network
    image_height : int
        Height of the input image to the network
    image_channels : int
        Number of channes of the input image to the network
    params : <NetworkParams object>
        Object collecting the hyperparameters and other variables of the network
    restore_index : int
        Checkpoint index to the model that has to be restored
    layers_to_optimise : list
        List of strings with the names of the layers to be optimized
    training : bool
        Flag indicating whether the network is going to be used for training or
        to do fordward passes
    ops_list : list
        List with tensorflow operations
    session : tf.Session()
        Tensorflow session used to run the operations
    is_restoring : bool
        Flag indicating whether the network model should be restored from a previous
        checkpoint
    restore_folder_conv : string
        Path to the folder with the convolutional part of the model to be restored
    restore_folder_fc_softmax : string
        Path to the fully connected layers of the model to be restored
    is_knowledge_transfers : bool
        Flag indicating whether the knowledge from a previously trained model should
        be used
    summary_op : tf.operation
        Operation the merge the Tensorflow summaries to be visualized in the tensorboard
    summary_writer_training : tf.operation
        Summary writer for the training
    summary_writer_validation : tf.operation
        Summary writer for the validation
    x_pl : tf.tensor
        Tensor placeholder for the input images [None, :attr:`image_width`, :attr:`image_height`, :attr:`image_channels`]
    y_target_pl : tf.tensor
        Tensor placeholder for the labels of the input images [None, :attr:`~network_params.number_of_animals`]
    keep_prob_pl : tf.tensor
        Tensor placeholder for the 'keep probability' in case of dropout
    loss_weights_pl : tf.tensor
        Tensor placeholder for the weights of the :attr:`loss`
    y_logits : tf.operation
        Output of the last fully connected layer
    softmax_probs : tf.operation
        Softmax probabilities computed from :attr:`y_logits`
    predictions : tf.operation
        Predictions of the given input images computed as the argmax of :attr:`softmax_probs`
    loss : tf.operation
        Loss function
    optimisation_step : tf.operation
        Optimization function
    global_step : tf.global_step
        Global tensorflow counter to save the checkpoints
    accuracy : tf.operation
        Accuracy of the :attr:`predictions` computed from the :attr:`y_logits`
    individual_accuracy : tf.operation
        Individual accuracy of the :attr:`predictions` for every class computed from the :attr:`y_logits`
    weights : ndarray
        Proportion of labels of every class in the collection of images in an epoch
    fc_vector : tf.operation
        Output of the second to last fully connected layer of the network
    saver_conv : tf.operation
        Operation to save the convolutional layers of the model
    saver_fc_softmax : tf.operation
        Operation to save the fully connected layers of the model
    save_folder_conv : string
        Path to the checkpoints of the convolutional layers of the model
    save_folder_fc_softmax : string
        Path to the checkpoints of the fully connected layers of th model

    """
    def __init__(self, params, training_flag = True, restore_index = None):
        # print("*** params", params.__dict__)
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
        knowledge_transfer_info_dict = {'input_image_size': self.params.target_image_size if self.params.target_image_size is not None else self.params.image_size,
                                'video_path': self.params.video_path,
                                'number_of_animals': self.params.number_of_animals,
                                'number_of_channels': self.params.number_of_channels,
                                'params': params}
        np.save(os.path.join(self.params.save_folder, 'info.npy'), knowledge_transfer_info_dict)
        self.set_savers()
        # Create list of operations to run during training and validation
        if self.training:
            self.ops_list = [self.loss, self.accuracy, self.individual_accuracy]
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            logger.debug('Restoring...')
            # Get subfolders from where we will load the network from previous checkpoints
            [self.restore_folder_conv, self.restore_folder_fc_softmax] = get_checkpoint_subfolders( self.params._restore_folder, ['conv', 'softmax'])
        elif self.is_knowledge_transfer:
            # Get subfolders from where we will load the convolutional filters to perform knowledge transfer
            logger.debug('Performing knowledge transfer...')
            [self.restore_folder_conv] = get_checkpoint_subfolders(self.params._knowledge_transfer_folder,['conv'])
            self.session.run(self.global_step.assign(0))
        if self.training:
            self.create_summaries_writers()

    @property
    def is_knowledge_transfer(self):
        if self.params._knowledge_transfer_folder is not None:
            self.restore_folder_fc_softmax = None
            logger.debug("restore_folder_fc_softmax: %s"  %self.restore_folder_fc_softmax)
        return self.params._knowledge_transfer_folder is not None

    @property
    def is_restoring(self):
        return self.params._restore_folder is not None

    # Create savers for the convolutions and the fully conected and softmax separately
    def set_savers(self):
        """Create or retrieves the paths where the checkpoint files will be saved and
        initialize the objects to save them

        See Also
        --------
        :func:`createSaver`
        """
        self.saver_conv = createSaver('saver_conv', exclude_fc_and_softmax = True)
        self.saver_fc_softmax = createSaver('saver_fc_softmax', exclude_fc_and_softmax = False)
        # Create subfolders where we will save the checkpoints of the trainig
        [self.save_folder_conv,self.save_folder_fc_softmax] = get_checkpoint_subfolders( self.params._save_folder, ['conv', 'softmax'])

    def create_summaries_writers(self):
        """Create the summary writers for thw tensorboard summaries
        """
        self.summary_op = tf.summary.merge_all()
        self.summary_writer_training = tf.summary.FileWriter(self.params._save_folder + '/train',self.session.graph)
        self.summary_writer_validation = tf.summary.FileWriter(self.params._save_folder + '/val',self.session.graph)

    def _build_graph(self):
        """Main tensorflow graph with the convolutional model and the operations
        needed for optimization during training

        See Also
        --------
        :meth:`weighted_loss`
        :meth:`set_optimizer`
        :meth:`evaluation`
        """
        self.x_pl = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_channels], name = 'images')
        logger.debug('training model %i' %self.params.cnn_model)

        model_image_width = self.image_width if self.params.target_image_size is None else self.params.target_image_size[0]
        model_image_height = self.image_height if self.params.target_image_size is None else self.params.target_image_size[1]
        logger.debug("plh image_width %s" %str(self.image_width))
        logger.debug("plh image_height %s" %str(self.image_height))
        logger.debug("model_image_width %s" %str(model_image_width))
        logger.debug("model_image_height %s" %str(model_image_height))
        self.y_logits, self.fc_vector = CNN_MODELS_DICT[self.params.cnn_model](self.x_pl,self.params.number_of_animals,
                                                                model_image_width, model_image_height,
                                                                self.image_channels)
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
        """Set layers of the network to be optimised during the training process
        """
        if self.params.scopes_layers_to_optimize is not None:
            self.layers_to_optimise = []
            for scope_layer in self.params.scopes_layers_to_optimize:
                with tf.variable_scope(scope_layer, reuse=True) as scope:
                    W = tf.get_variable("weights")
                    B = tf.get_variable("biases")
                self.layers_to_optimise.append([W, B])
        self.layers_to_optimise = [layer for layers in self.layers_to_optimise for layer in layers]

    def weighted_loss(self):
        """Loss function minimised during the training process
        """
        cross_entropy = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(self.y_target_pl, self.y_logits, self.loss_weights_pl), name = 'CrossEntropyMean')
        self._add_loss_summary(cross_entropy)
        return cross_entropy

    def compute_loss_weights(self, training_labels):
        """Computes the label weights for all the images that will be train in an epoch
        """
        self.weights = 1. - np.sum(training_labels,axis=0) / len(training_labels)

    def set_optimizer(self):
        """Sets the optimization operation to be perform during the training process

        See Also
        --------
        :meth:`get_layers_to_optimize`

        """
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
        """Computes and returns the accuracy and individual accuracy of the predictions
        given by the model for the input images

        See Also
        --------
        :func:`compute_individual_accuracy`
        :func:`compute_accuracy`
        """
        individual_accuracy = compute_individual_accuracy(self.y_target_pl, self.y_logits, self.params.number_of_animals)
        accuracy = compute_accuracy(self.y_target_pl, self.y_logits)
        return accuracy, individual_accuracy

    def reinitialize_softmax_and_fully_connected(self):
        """Reinizializes the weights in the two last fully connected layers
        of the network
        """
        logger.debug('Reinitializing softmax and fully connected')
        self.session.run(tf.variables_initializer(
            [v for v in tf.global_variables()
             if 'soft' in v.name or 'full' in v.name]))

    # def reinitialize_conv_layers(self):
    #     """Reinizializes the weights in the conv layers indicated in the list
    #     conv_layers_list. The list can include the names 'conv1', 'conv2' or
    #     'conv3'
    #     """
    #     if self.params.kt_conv_layers_to_discard is not None:
    #         logger.debug('Reinitializing conv layers')
    #         logger.debug(self.params.kt_conv_layers_to_discard)
    #         self.session.run(tf.variables_initializer(
    #             [v for v in tf.global_variables()
    #              if v.name in self.params.kt_conv_layers_to_discard]))

    @staticmethod
    def check_checkpoint_path(checkpoint_path, restore_folder):
        if os.path.isfile(checkpoint_path + '.meta'):
            return checkpoint_path
        elif not os.path.isfile(checkpoint_path + '.meta'):
            ckpt = os.path.split(checkpoint_path)[1]
            return os.path.join(restore_folder, ckpt)

    def restore_convolutional_layers(self):
        """Restores the weights of the convolutional layers from a checkpoint file
        of a previously trained model

        See Also
        --------
        :meth:`saver_conv`
        """
        ckpt = tf.train.get_checkpoint_state(self.restore_folder_conv)
        if self.restore_index is None:
            ckpt_path = self.check_checkpoint_path(ckpt.model_checkpoint_path, self.restore_folder_conv)
            logger.debug('Restoring convolutional part from %s' % ckpt_path)
            self.saver_conv.restore(self.session, ckpt_path) # restore convolutional variables
        else:
            logger.debug('Restoring convolutional part from %s' %ckpt.all_model_checkpoint_paths[self.restore_index])
            self.saver_conv.restore(self.session, ckpt.all_model_checkpoint_paths[self.restore_index])


    def restore_classifier(self):
        """Restores the weights of the convolutional layers from a checkpoint file
        of a previously trained model

        See Also
        --------
        :meth:`saver_fc_softmax`
        """
        ckpt = tf.train.get_checkpoint_state(self.restore_folder_fc_softmax)
        if self.restore_index is None:
            self.saver_fc_softmax.restore(self.session, ckpt.model_checkpoint_path) # restore fully-conected and softmax variables
            logger.debug('Restoring fully-connected and softmax part from %s' %ckpt.model_checkpoint_path)
        else:
            self.saver_fc_softmax.restore(self.session, ckpt.all_model_checkpoint_paths[self.restore_index]) # restore fully-conected and softmax variables
            logger.debug('Restoring fully-connected and softmax part from %s' %ckpt.all_model_checkpoint_paths[self.restore_index])

    def restore(self):
        """Restores a previously trained model from a checkpoint folder

        See Also
        --------
        :meth:`restore_convolutional_layers`
        :meth:`restore_classifier`
        """
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            self.restore_convolutional_layers()
            self.restore_classifier()
        elif self.is_knowledge_transfer:
            self.restore_convolutional_layers()

    def compute_batch_weights(self, batch_labels):
        """Returns the weights vector (`batch_weights`) for a batch of labes
        """
        batch_weights = np.sum(self.weights*batch_labels,axis=1)
        return batch_weights

    def get_feed_dict(self, batch):
        """Returns the dictionary `feed_dict` with the variables and parameters
        needed to perform certain operations for a give `batch`

        See Also
        --------
        :meth:`compute_batch_weights`

        """
        (batch_images, batch_labels) = batch
        batch_weights = self.compute_batch_weights(batch_labels)
        return  { self.x_pl: batch_images,
                  self.y_target_pl: batch_labels,
                  self.loss_weights_pl: batch_weights,
                  self.keep_prob_pl: self.params.keep_prob}

    def train(self,batch):
        """Runs in the tensorflow session :attr:`session` an :attr:`optimization_step`
        and the operations in :attr:`ops_list` for a given `batch`

        See Also
        --------
        :meth:`get_feed_dict`
        """
        feed_dict = self.get_feed_dict(batch)
        self.session.run(self.optimisation_step,feed_dict = feed_dict)
        outList = self.session.run(self.ops_list, feed_dict = feed_dict)
        return outList, feed_dict

    def validate(self,batch):
        """Runs in the tensorflow session :attr:`session` the operations
        in :attr:`ops_list` for a given `batch`

        See Also
        --------
        :meth:`get_feed_dict`
        """
        feed_dict = self.get_feed_dict(batch)
        outList = self.session.run(self.ops_list, feed_dict = feed_dict)
        return outList, feed_dict

    def predict(self,batch):
        """Runs in the tensorflow session :attr:`session` the :attr:`softmax_probs`
        and the :attr:`predictions` operations
        """
        feed_dict = {self.x_pl: batch}
        return self.session.run([self.softmax_probs,self.predictions], feed_dict = feed_dict)

    # def get_fully_connected_vectors(self,batch):
    #     """Runs in the tensorflow session :attr:`session` the :attr:`fc_vector`
    #     operation
    #     """
    #     feed_dict = {self.x_pl: batch}
    #     return self.session.run(self.fc_vector, feed_dict = feed_dict)

    def write_summaries(self,epoch_i,feed_dict_train, feed_dict_val):
        """Writes the summaries using the :attr:`summary_str_training` and
        :attr:`summary_writer_validation` to be visualized in the Tensorboard
        """
        summary_str_training = self.session.run(self.summary_op, feed_dict=feed_dict_train)
        summary_str_validation = self.session.run(self.summary_op, feed_dict=feed_dict_val)
        self.summary_writer_training.add_summary(summary_str_training, epoch_i)
        self.summary_writer_validation.add_summary(summary_str_validation, epoch_i)

    def _add_loss_summary(self,loss):
        """Adds the :attr:`loss` in the summaries for the Tensorboard
        """
        tf.summary.scalar(loss.op.name, loss)

    def save(self):
        """Saves the models in the correspoding :attr:`save_folder_conv` and :attr:`save_folder_fc_softmax` folders
        using the saver :attr:`saver_conv` and :attr:`saver_fc_softmax`
        """
        self.saver_conv.save(self.session, os.path.join(self.save_folder_conv, "conv.ckpt"), global_step = self.global_step)
        self.saver_fc_softmax.save(self.session, os.path.join(self.save_folder_fc_softmax, "softmax.ckpt"), global_step = self.global_step)

def compute_individual_accuracy(labels, logits, classes):
    """Computes the individual accuracy for a set of `labels` and `logits` given the
    number of `classes`

    Parameters
    ----------
    labels : tf.tensor
        Tensor of shape [batch size, number of classes] with the labels of the input
        images in dense format
    logits : tf.tensor
        Tensor of shape [batch size, number of classes] with the output of the last
        fully connected layer
    classes : int
        Number of classes

    Returns
    -------
    individual_accuracies : tf.tensor
        Tensor of shape [number of classes] with the accuracy for every class
    """
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
    individual_accuracies = tf.div(countCorrect,numImagesPerIndiv)
    return individual_accuracies

def compute_accuracy(labels, logits):
    """Computes the accuracy for a set of `labels` and `logits`

    Parameters
    ----------
    labels : tf.tensor
        Tensor of shape [batch size, number of classes] with the labels of the input
        images in dense format
    logits : tf.tensor
        Tensor of shape [batch size, number of classes] with the output of the last
        fully connected layer

    Returns
    -------
    accuracy : tf.constant
        Accuracy for the given set of `logits` correspoding to the `labels`
    """
    # We add 1 to the labels and predictions to avoid having a 0 label
    labels = tf.cast(tf.add(tf.where(tf.equal(labels,1))[:,1],1),tf.float32)
    predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float32)
    # acc = tf.metrics.accuracy(labels, predictions)
    correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
    return accuracy


def get_checkpoint_subfolders(folder_name, sub_folders_names):
    """
    Create if it does not exist the folder `folder_name` and the subfolders
    `sub_folders_names`. It returns the list of `sub_paths` created or
    """
    sub_paths = []
    for name in sub_folders_names:
        subPath = folder_name + '/' + name
        if not os.path.exists(subPath):
            os.makedirs(subPath)
            logger.debug('%s has been created' %subPath)
        else:
            logger.debug('%s already exists' %subPath)
        sub_paths.append(subPath)
    return sub_paths

def createSaver(name, exclude_fc_and_softmax):
    """Returns a tensorflow `saver` with a given `name` which will only save the
    convolutional layers if `exclude_fc_and_softmax` is True, and it will only save
    the fully connected layers if `exclude_fc_and_softmax` is False
    """
    if not exclude_fc_and_softmax:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' in v.name or 'full' in v.name], name = name, max_to_keep = 1000000)
    elif exclude_fc_and_softmax:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'conv' in v.name], name = name, max_to_keep = 1000000)
    else:
        raise ValueError('The second argument has to be a boolean')

    return saver
