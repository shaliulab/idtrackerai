import tensorflow as tf
import os
from cnn_architectures import cnn_model_1

IMAGE_SIZE = (32,32,1)

class ConvNetwork():
    def __init__(self,
                image_size = IMAGE_SIZE,
                number_of_animals = None,
                learning_rate = None,
                keep_prob = None,
                restore_folder = None,
                save_folder = None,
                knowledge_transfer_folder = None,
                use_adam_optimiser = False,
                scopes_layers_to_optimize = None ):
        # Set main attibutes of the class
        self.image_width = IMAGE_SIZE[0]
        self.image_height = IMAGE_SIZE[1]
        self.image_channels = IMAGE_SIZE[2]
        self.number_of_animals = number_of_animals
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.restore_folder = restore_folder # folder where we are going to restore the state of the network from a previous check point
        self.save_folder = save_folder # folder where we are going to save the checkpoints of the current training
        self.kt_folder = knowledge_transfer_folder # folder where we are going to load the network to perform knowledge transfer
        self.use_adam_optimiser = use_adam_optimiser
        self.scopes_layers_to_optimize = scopes_layers_to_optimize
        # Initialize layers to optimize to be empty. This means tha we will
        # optimize all the layers of our network
        self._layers_to_optimise = []
        # Build graph with the network, loss, optimizer and accuracies
        self.x_pl, \
        self.y_target_pl,\
        self.y_logits, \
        self.loss_weights_pl, \
        self.global_step, \
        loss, \
        optimisation_step, \
        accuracy, \
        individual_accuracy = self._build_graph()
        # Create savers for the convolutions and the fully conected and softmax separately
        self.saver_conv = createSaver('saver_conv', False)
        self.saver_fc_softmax = createSaver('saver_fc_softmax', True)
        # Create list of operations to run during training and validation
        self.train_ops_list = [optimisation_step, loss, accuracy, individual_accuracy]
        self.val_ops_list = [loss, accuracy, individual_accuracy]
        # Create subfolders where we will save the checkpoints of the trainig
        [self.save_folder_conv,self.save_folder_fc_softmax] = create_checkpoint_subfolders( self.save_folder, ['conv', 'softmax'])

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if self.is_restoring:
            # Get subfolders from where we will load the network from previous checkpoints
            [self.restore_folder_conv,self.restore_folder_fc_softmax] = get_checkpoint_subfolders( self.restore_folder, ['conv', 'softmax'])
        elif self.is_knowledge_transfer:
            # Get subfolders from where we will load the convolutional filters to perform knowledge transfer
            [self.restore_folder_conv] = get_checkpoint_subfolders(self.kt_folder,['conv'])
            self.global_step.assign(0).eval()
        self.restore()

        self.summary_op = tf.summary.merge_all()
        self.summary_writer_training = tf.summary.FileWriter(self.save_folder + '/train',self.session.graph)
        self.summary_writer_validation = tf.summary.FileWriter(self.save_folder + '/val',self.session.graph)

    @property
    def is_knowledge_transfer():
        if self.kt_folder is not None:
            self.restore_folder_fc_softmax = None
        return self.kt_folder is not None

    @property
    def is_restoring():
        return self.restore_folder is not None

    def _build_graph(self):
        x_pl = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_channels], name = 'images')
        y_target_pl = tf.placeholder(tf.int32, [None, self.number_of_animals], name = 'labels')
        loss_weights_pl = tf.placeholder(tf.float32, [None], name = 'loss_weights')

        y_logits = cnn_model_1(x_pl)

        loss = self.weighted_loss(y_target_pl, y_logits, loss_weights_pl)
        optimisation_step, global_step = self.set_optimizer(loss, self.learning_rate)
        accuracy, individual_accuracy = evaluation(y_target_pl, y_logits, self.number_of_animals)

        return x_pl, y_target_pl, y_logits, loss_weights_pl, loss, optimisation_step, global_step, accuracy, individual_accuracy

    def get_layers_to_optimize(self):
        if self.scopes_layers_to_optimize is not None:
            for scope_layer in self.scopes_layers_to_optimize:
                with tf.variable_scope(scope_layer, reuse=True) as scope:
                    W = tf.get_variable("weights")
                    B = tf.get_variable("biases")
                self.layers_to_optimise.append([W, B])
        self.layers_to_optimise = [layer for layer in layers for layers in self.layers_to_optimise]

    def weighted_loss(y, y_logits, loss_weights):
        cross_entropy = tf.reduce_mean(
            tf.contrib.losses.softmax_cross_entropy(y_logits,y, loss_weights), name = 'CrossEntropyMean')
        _add_loss_summary(cross_entropy)
        return cross_entropy

    def compute_loss_weights(self, training_labels):
        self.weights = 1. - np.sum(labels,axis=0) / len(labels)

    def set_optimizer(self, loss):
        if not self.use_adam_optimiser:
            print 'Training with SGD'
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.use_adam_optimiser:
            print 'Training with ADAM'
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.scopes_layers_to_optimize is not None:
            self.get_layers_to_optimize()
            train_op = optimizer.minimize(loss, var_list = self.layers_to_optimise)
        else:
            train_op = optimizer.minimize(loss=loss)
        return train_op, global_step

    def evaluation(y,y_logits):
        individual_accuracy = compute_individual_accuracy(y,y_logits, self.number_of_animals)
        accuracy = compute_accuracy(y,y_logits)
        return accuracy, indivAcc

    def restore(self):
        conv_ckpt = tf.train.get_checkpoint_state(self.restore_folder_conv)
        try:
            self.saver_conv.restore(self.session, ckpt.conv_checkpoint_path) # restore convolutional variables
        except:
            print('Warning: no checkpoints found for the convolutional part')
        fc_softmax_ckpt = tf.train.get_checkpoint_state(self.restore_folder_fc_softmax)
        if self.is_restoring:
            try:
                self.saver_fc_softmax.restore(self.session, ckpt.model_checkpoint_path) # restore fully-conected and softmax variables
            except:
                print('Warning: no checkpoints found for the fully-connected and softmax parts')

    def compute_batch_weights(self, batch_labels):
        if self.weighted_flag:
            batch_weights = np.sum(weights*labels_feed,axis=1)
        elif not self.weighted_flag:
            batch_weights = np.ones(len(labels_feed))

        return batch_weights

    def train(self,batch,epoch_i):
        (batch_images, batch_labels) = batch
        batch_weights = compute_batch_weights(self, batch_labels)
        feed_dict = {
          images_pl: batch_images,
          labels_pl: batch_labels,
          loss_weights_pl: batch_weights,
          keep_prob_pl: self.keep_prob
        }
        outList = self.session.run(self.train_ops_list, feed_dict=feed_dict)
        outList.append(feed_dict)

        return outList

    def validate(self,batch):
        (batch_images, batch_labels) = batch
        feed_dict = {
          images_pl: batch_images,
          labels_pl: batch_labels,
          keep_prob_pl: 1.0
        }
        outList = self.session.run(self.val_ops_list, feed_dict=feed_dict)
        outList.append(feed_dict)

        return outList

    def write_summaries(self,epoch_i,feed_dict_train, feed_dict_val):
        summary_str_training = self.sess.run(summary_op, feed_dict=feed_dict_train)
        summary_str_validation = self.sess.run(summary_op, feed_dict=feed_dict_val)
        self.summary_writer_training.add_summary(summary_str_training, epoch_i)
        self.summary_writer_validation.add_summary(summary_str_validation, epoch_i)

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
    predictions = tf.cast(tf.add(tf.argmax(logits,1),1),tf.float)
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
            print subPath + ' has been created'
        else:
            print subPath + ' already exists'
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
            print subPath + ' has been created'
        else:
            print subPath + ' already exists'
        subPaths.append(subPath)
    return subPaths

def createSaver(name, exclude_fc_and_softmax):
    if include:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' in v.name or 'full' in v.name], name = name)
    elif not include:
        saver = tf.train.Saver([v for v in tf.global_variables() if 'soft' not in v.name or 'full' not in v.name], name = name)
    else:
        raise ValueError('The second argument has to be a boolean')

    return saver
