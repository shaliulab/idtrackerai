from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
import numpy as np
import sys
sys.path.append('./network')
sys.path.append('./network/identification_model')
sys.path.append('./utils')
from id_CNN import ConvNetwork
from network_params import NetworkParams
from get_data import DataSet
from get_predictions import GetPrediction

class EmbeddingVisualiser(object):
    def __init__(self, labels = None, features = None):
        if labels is not None:
            self.labels0 = labels[0]
            self.labels1 = labels[1]
        if features is not None:
            self.embedding_var0 = tf.Variable(features[0], name = 'features0_fc8')
            self.embedding_var1 = tf.Variable(features[1], name = 'features0_fc100')
            self.embedding_var2 = tf.Variable(features[2], name = 'features1_fc8')
            self.embedding_var3 = tf.Variable(features[3], name = 'features1_fc100')

    def create_labels_file(self, embeddings_folder):
        self.labels0_path = os.path.join(embeddings_folder, 'labels0.csv')
        self.labels1_path = os.path.join(embeddings_folder, 'labels1.csv')
        df = pd.DataFrame(self.labels0)
        df.to_csv(self.labels0_path, sep='\t')
        df = pd.DataFrame(self.labels1)
        df.to_csv(self.labels1_path, sep='\t')


    def create_sprite_file(self, images, labels):
        """
        Generates sprite image and associated labels
        """
        sprite_width = sprite_height = self.image_size[0] * 256
        images_first_gf, labels_first_gf = list_of_global_fragments.global_fragments[0].get_images_and_labels()
        images1, labels1 = list_of_global_fragments.global_fragments[1].get_images_and_labels()
        number_of_columns = 256
        number_of_rows = 256
        number_of_images = int(number_of_columns * number_of_rows)
        numImagesPerIndiv = int(np.floor(number_of_images / num_indiv_to_represent))

        linearImages = np.reshape(images, [number_of_images, self.image_size[0] ** 2])
        imagesTSNE = []

        for ind in range(num_indiv_to_represent):
            imagesTSNE.append(images[ind][:numImagesPerIndiv])

        rowSprite = []
        sprite = []
        i = 0

        while i < number_of_images:
            rowSprite.append(images[i])

            if (i+1) % number_of_columns == 0:
                sprite.append(np.hstack(rowSprite))
                rowSprite = []
            i += 1

        sprite = np.vstack(sprite)
        spriteName = str(num_indiv_to_represent) + '_fish_'+ str(number_of_images)+'imgs_sprite.png'
        cv2.imwrite(spriteName, uint8caster(sprite))

        imageName = str(num_indiv_to_represent) + '_fish_'+ str(number_of_images)+'images.pkl'
        pickle.dump(linearImages, open(imageName, "wb"))

        labelName = str(num_indiv_to_represent) + '_fish_'+ str(number_of_images)+'labels.tsv'
        df = pd.DataFrame(labels)
        df.to_csv(labelName, sep='\t')

        return images, labels

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
            embedding2 = config.embeddings.add()
            embedding2.tensor_name = self.embedding_var2.name
            embedding3 = config.embeddings.add()
            embedding3.tensor_name = self.embedding_var3.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding0.metadata_path =  self.labels0_path
            embedding1.metadata_path =  self.labels0_path
            embedding2.metadata_path =  self.labels1_path
            embedding3.metadata_path =  self.labels0_path
            # Saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(summary_writer, config)
            step.assign(0).eval()
            saver.save(session, os.path.join(checkpoint_folder, "model.ckpt"), step)

def visualize_embeddings_global_fragments(video, list_of_global_fragments, list_of_fragments, params, print_flag):
    net = ConvNetwork(params, training_flag = False)
    try:
        imagesT, labelsT = list_of_global_fragments.global_fragments[0].get_images_and_labels()
        imagesV, labelsV = list_of_global_fragments.global_fragments[1].get_images_and_labels()
    except:
        list_of_global_fragments.relink_fragments_to_global_fragments(list_of_fragments.fragments)
        imagesT, labelsT = list_of_global_fragments.global_fragments[0].get_images_and_labels()
        imagesV, labelsV = list_of_global_fragments.global_fragments[1].get_images_and_labels()
    imagesT = np.asarray(imagesT)
    imagesV = np.asarray(imagesV)
    if len(imagesT[0].shape) < 3:
        imagesT = np.expand_dims(imagesT, axis = 3)
        imagesV = np.expand_dims(imagesV, axis = 3)
    dataT = DataSet(params.number_of_animals, imagesT)
    dataV = DataSet(params.number_of_animals, imagesV)
    # Restore network
    net.restore()
    # Train network
    assignerT = GetPrediction(dataT)
    assignerV = GetPrediction(dataV)
    # Get fully connected vectors
    assignerT.get_predictions_fully_connected_embedding(net.get_fully_connected_vectors, video.number_of_animals)
    assignerV.get_predictions_fully_connected_embedding(net.get_fully_connected_vectors, video.number_of_animals)
    assignerT.get_predictions_softmax(net.predict)
    assignerV.get_predictions_softmax(net.predict)
    print(assignerV._fc_vectors)
    # Visualize embeddings
    video.create_embeddings_folder()
    visualize_fully_connected_embedding = EmbeddingVisualiser(labels = [labelsT, labelsV],
                                                            features = [assignerT.softmax_probs,
                                                                        assignerT._fc_vectors,
                                                                        assignerV.softmax_probs,
                                                                        assignerV._fc_vectors])
    visualize_fully_connected_embedding.create_labels_file(video.embeddings_folder)
    visualize_fully_connected_embedding.visualize(video.embeddings_folder)
    return assignerT

if __name__ == "__main__":
    video = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/video_object.npy').item()
    list_of_global_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/preprocessing/global_fragments.npy').item()
    list_of_fragments = np.load('/home/lab/Desktop/TF_models/IdTrackerDeep/videos/8zebrafish_conflicto/session_20171207/preprocessing/fragments.npy').item()
    params = NetworkParams(video.number_of_animals,
                                learning_rate = 0.005,
                                keep_prob = 1.0,
                                scopes_layers_to_optimize = None,
                                save_folder = video.accumulation_folder,
                                restore_folder = video.accumulation_folder,
                                image_size = video.identification_image_size,
                                video_path = video.video_path)
    a = visualize_embeddings_global_fragments(video, list_of_global_fragments, list_of_fragments, params, True)
