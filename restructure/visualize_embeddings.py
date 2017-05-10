from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd

class EmbeddingVisualiser(object):
    def __init__(self, labels = None, features = None):
        self.labels0 = labels[0]
        self.labels1 = labels[1]
        if features is not None:
            self.embedding_var0 = tf.Variable(features[0], name='features0')
            self.embedding_var1 = tf.Variable(features[1], name='features1')

    def create_labels_file(self, embeddings_folder):
        self.labels0_path = os.path.join(embeddings_folder,'labels0.csv')
        self.labels1_path = os.path.join(embeddings_folder,'labels1.csv')
        df = pd.DataFrame(self.labels0)
        df.to_csv(self.labels0_path, sep='\t')
        df = pd.DataFrame(self.labels1)
        df.to_csv(self.labels1_path, sep='\t')

    def visualize(self, embeddings_folder):
        step = tf.Variable(0, name='step', trainable=False)
        saver = tf.train.Saver()
        checkpoint_folder = os.path.join(embeddings_folder,'checkpoints')
        if os.path.isdir(checkpoint_folder) == False:
            os.makedirs(checkpoint_folder)

        with tf.Session() as session:
            tf.global_variables_initializer().run()
            # Use the same checkpoint_folder where you stored your checkpoint.
            summary_writer = tf.summary.FileWriter(checkpoint_folder)
            # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
            config = projector.ProjectorConfig()
            # You can add multiple embeddings. Here we add only one.
            embedding0 = config.embeddings.add()
            embedding0.tensor_name = self.embedding_var0.name
            embedding1 = config.embeddings.add()
            embedding1.tensor_name = self.embedding_var1.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding0.metadata_path =  self.labels0_path
            embedding1.metadata_path =  self.labels1_path
            # Saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            step.assign(0).eval()
            saver.save(session, os.path.join(checkpoint_folder, "model.ckpt"), step)

def visualize_embeddings_global_fragments(video, global_fragments, params, print_flag):
    net = ConvNetwork(params, training_flag = False)
    # Get images from the blob collection
    imagesT, labelsT = get_images_and_labels_from_global_fragment(global_fragments[0])
    imagesV, labelsV = get_images_and_labels_from_global_fragment(global_fragments[1])
    # build data object
    imagesT = np.expand_dims(np.asarray(imagesT), axis = 3)
    imagesV = np.expand_dims(np.asarray(imagesV), axis = 3)
    dataT = DataSet(params.number_of_animals, imagesT)
    dataV = DataSet(params.number_of_animals, imagesV)
    # Instantiate data_set
    dataT.standarize_images()
    dataV.standarize_images()
    # Crop images from 36x36 to 32x32 without performing data augmentation
    dataT.crop_images(image_size = 32)
    dataV.crop_images(image_size = 32)
    # Restore network
    net.restore()
    # Train network
    assignerT = GetPrediction(dataT, print_flag = print_flag)
    assignerV = GetPrediction(dataV, print_flag = print_flag)
    # Get fully connected vectors
    assignerT.get_predictions_fully_connected_embedding(net.get_fully_connected_vectors, video.number_of_animals)
    assignerV.get_predictions_fully_connected_embedding(net.get_fully_connected_vectors, video.number_of_animals)
    # Visualize embeddings
    video.create_embeddings_folder()
    visualize_fully_connected_embedding = EmbeddingVisualiser(labels = [labelsT, labelsV],
                                                            features = [assignerT._fc_vectors, assignerV._fc_vectors])
    visualize_fully_connected_embedding.create_labels_file(video._embeddings_folder)
    visualize_fully_connected_embedding.visualize(video._embeddings_folder)
