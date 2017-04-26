import tensorflow as tf
import os

IMAGE_SIZE = (32,32,1)
class ConvNetwork():
    def __init__(self, number_of_animals, learning_rate, keep_prob, from_checkpoint_path = None, image_size = IMAGE_SIZE, use_adam_optimiser = False, scopes_layers_to_optimize = None ):
        self.image_width = IMAGE_SIZE[0]
        self.image_height = IMAGE_SIZE[1]
        self.image_channels = IMAGE_SIZE[2]
        self.number_of_animals = number_of_animals
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.use_adam_optimiser = use_adam_optimiser
        self.scopes_layers_to_optimize = scopes_layers_to_optimize
        self._layers_to_optimise = []
        self.x_pl, \
        self.y_target_pl,\
        self.y_logits, \
        self.loss_weights_pl, \
        self.global_step, \
        loss, \
        optimisation_step, \
        accuracy, \
        individual_accuracy = self._build_graph()
        self.train_ops_list = [optimisation_step, loss, accuracy, individual_accuracy]
        self.val_ops_list = [loss, accuracy, individual_accuracy]

        [ckpt_dir_model,ckpt_dir_softmax] = createCkptFolder( from_checkpoint_path, ['model', 'softmax'])

        self.session = tf.Session()

        if from_checkpoint_path == None:
            self.session.run(tf.global_variables_initializer())
        else:
            self.restore(from_checkpoint_path)

    def _build_graph(self):
        x_pl = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, self.image_channels], name = 'images')
        y_target_pl = tf.placeholder(tf.int32, [None, self.number_of_animals], name = 'labels')
        loss_weights_pl = tf.placeholder(tf.float32, [None], name = 'loss_weights')

        conv1 = tf.contrib.layers.conv2d(
            x_pl, 16, 5, 1,
            activation_fn=tf.nn.relu,
            padding='SAME',
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0),
            biases_initializer = tf.constant_initializer(0.0),
            scope="conv1")
        maxpool1 = tf.contrib.layers.max_pool2d(
            conv1,
            kernel_size = 2,
            stride = 2,
            padding = 'SAME',
            scope = "maxpool1")
        conv2 = tf.contrib.layers.conv2d(
            X, 64, 5, 1,
            activation_fn=tf.nn.relu,
            padding='SAME',
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0),
            biases_initializer = tf.constant_initializer(0.0),
            scope="conv2")
        maxpool2 = tf.contrib.layers.max_pool2d(
            conv1,
            kernel_size = 2,
            stride = 2,
            padding = 'SAME',
            scope = "maxpool2")
        conv3 = tf.contrib.layers.conv2d(
            X, 100, 5, 1,
            activation_fn=tf.nn.relu,
            padding='SAME',
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0),
            biases_initializer = tf.constant_initializer(0.0),
            scope="conv3")
        conv3_flat = tf.reshape(conv3, [-1, np.prod(conv3.shape[1:])], name = 'conv5_reshape')
        fc1 = tf.contrib.layers.fully_connected(
            conv3_flat, 100,
            activation_fn = tf.nn.relu,
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0),
            biases_initializer = tf.constant_initializer(0.0),
            scope="fully-connected1")
        y_logits = tf.contrib.layers.fully_connected(
            fc1, self.number_of_animals,
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=0),
            biases_initializer = tf.constant_initializer(0.0),
            scope = "logits")

        loss = self.weighted_loss(y_target_pl, y_logits, loss_weights_pl)
        optimisation_step, global_step = self.optimize(loss, self.learning_rate)
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

    def optimize(self, loss):
        """Choose the optimiser to be used and the layers to be optimised
        :param loss: the function to be minimised
        :param lr: learning rate
        :param layer_to_optimise: layers to be trained (in tf representation)
        :param use_adam: if True uses AdamOptimizer else SGD
        """
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
        accuracy, individual_accuracy = compute_individual_accuracy(y,y_logits, self.number_of_animals)
        return accuracy, indivAcc

    def compute_loss_weights(self, training_labels):
        self.weights = 1. - np.sum(labels,axis=0) / len(labels)

    def compute_batch_weights(self, batch_labels):
        if self.weighted_flag:
            batch_weights = np.sum(weights*labels_feed,axis=1)
        elif not self.weighted_flag:
            batch_weights = np.ones(len(labels_feed))

        return batch_weights

    def train(self,batch):
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

    def save(self, global_step):
        self.saver_model.save(self.session, os.path.join(self.ckpt_dir_model, "model.ckpt"), global_step = global_step)
        self.saver_softmax.save(self.session, os.path.join(self.ckpt_dir_softmax, "softmax.ckpt"), global_step = global_step)

    def knowledge_transfer(self, model_checkpoint_path):
        restore_from_folder(model_checkpoint_path, self.model_saver, self.session)

    def restore(self, from_chekcpoint_path):
        self.session.run(tf.global_variables_initializer())
        # Load weights from a pretrained model if there is not any model saved
        # in the ckpt folder of the test
        ckpt = tf.train.get_checkpoint_state(ckpt_dir_model)
        if (not (ckpt and ckpt.model_checkpoint_path)) and accumCounter == 0:
            if loadCkpt_folder:
                if printFlag:
                    print '********************************************************'
                    print 'We are only loading the model'
                    print '********************************************************'
                loadCkpt_folder = loadCkpt_folder + '/model'

                if printFlag:
                    print 'loading weigths from ' + loadCkpt_folder

                restoreFromFolder(loadCkpt_folder, saver_model, sess)
                global_step.assign(0).eval()

            else:
                warnings.warn('It is not possible to perform knowledge transfer, give a folder containing a trained model')
        else:
            loadCkpt_folder_model = loadCkpt_folder + '/model'
            loadCkpt_folder_softmax = loadCkpt_folder + '/softmax'

            if printFlag:
                print "\n"
                print '********************************************************'
                print 'We are also loading the softmax'
                print '********************************************************'
                print 'loading weigths from ' + loadCkpt_folder + '/model'
                print 'loading softmax from ' + loadCkpt_folder + '/softmax'
            restoreFromFolder(loadCkpt_folder_model, saver_model, sess)
            restoreFromFolder(loadCkpt_folder_softmax, saver_softmax, sess)




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

    correct_prediction = tf.equal(predictions, labels, name='correctPrediction')
    acc = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='overallAccuracy')
    # acc = tf.reduce_mean(indivAcc)

    return acc,indivAcc

def getCkptvideoPath(videoPath, accumCounter, train=0):
    def getLastSession(subFolders):
        if len(subFolders) == 0:
            lastIndex = 0
        else:
            subFolders = natural_sort(subFolders)[::-1]
            lastIndex = int(subFolders[0].split('_')[-1])
        return lastIndex

    video = os.path.basename(videoPath)
    folder = os.path.dirname(videoPath)
    filename, extension = os.path.splitext(video)
    subFolder = folder + '/CNN_models'
    subSubFolders = glob.glob(subFolder +"/*")
    lastIndex = getLastSession(subSubFolders)
    sessionPath = subFolder + '/Session_' + str(lastIndex)

    if accumCounter >= 0:
        ckptvideoPath = sessionPath + '/AccumulationStep_' + str(accumCounter)
    elif accumCounter < 0:
        ckptvideoPath = sessionPath + '/pre_training'
    return ckptvideoPath

def createCkptFolder(folderName, subfoldersNameList):
    '''
    create if it does not exist the folder folderName in CNN and
    the same for the subfolders in the subfoldersNameList
    '''
    if not os.path.exists(folderName): # folder does not exist
        os.makedirs(folderName) # create a folder
        print folderName + ' has been created'
    else:
        print folderName + ' already exists'

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

def restore_from_folder(path_to_ckpt, saver, session):
    '''
    Restores variables stored in path_to_ckpt with a certain saver,
    for a particular (TF) session
    '''
    ckpt = tf.train.get_checkpoint_state(path_to_ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        print "restoring from " + ckpt.model_checkpoint_path
        saver.restore(session, ckpt.model_checkpoint_path) # restore model variables
