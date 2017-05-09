from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd

class EmbeddingVisualiser(object):
    def __init__(self, labels = None, features = None):
        self.labels = labels
        if features is not None:
            self.embedding_var = tf.Variable(features, name='features')

    def create_labels_file(self, embeddings_folder):
        self.labels_path = os.path.join(embeddings_folder,'labels.csv')
        df = pd.DataFrame(self.labels)
        df.to_csv(self.labels_path, sep='\t')

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
            embedding = config.embeddings.add()
            embedding.tensor_name = self.embedding_var.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path =  self.labels_path
            # Saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            step.assign(0).eval()
            saver.save(session, os.path.join(checkpoint_folder, "model.ckpt"), step)
