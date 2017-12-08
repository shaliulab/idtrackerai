from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd

class EmbeddingVisualiser(object):
    def __init__(self, video, labels = None, features = None):
        self.labels0 = labels[0]
        self.labels1 = labels[1]
        self.number_of_animals = video.number_of_animals
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

    def create_sprite_file(sprite_width = 8192, sprite_height = 8192, num_indiv_to_represent = self.number_of_animals):
        """
        Generates sprite image and associated labels
        """


        indIndices = np.random.permutation(60)
        indIndices = indIndices[:num_indiv_to_represent]
        images, labels = getImagesAndLabels(imdbTrain, indIndices)
        imH, imW = images[0][0].shape
        numColumns = sprite_width / imW
        numRows = sprite_height / imH
        numImages = int(numColumns * numRows)
        numImagesPerIndiv = int(np.floor(numImages / num_indiv_to_represent))

        images, labels = prepareTSNEImages(images, labels,numImagesPerIndiv)
        linearImages = np.reshape(images, [numImages, imH*imW])

        imagesTSNE = []

        for ind in range(num_indiv_to_represent):
            imagesTSNE.append(images[ind][:numImagesPerIndiv])

        rowSprite = []
        sprite = []
        i = 0

        while i < numImages:
            rowSprite.append(images[i])

            if (i+1) % numColumns == 0:
                sprite.append(np.hstack(rowSprite))
                rowSprite = []
            i += 1

        sprite = np.vstack(sprite)
        spriteName = str(num_indiv_to_represent) + '_fish_'+ str(numImages)+'imgs_sprite.png'
        cv2.imwrite(spriteName, uint8caster(sprite))

        imageName = str(num_indiv_to_represent) + '_fish_'+ str(numImages)+'images.pkl'
        pickle.dump(linearImages, open(imageName, "wb"))

        labelName = str(num_indiv_to_represent) + '_fish_'+ str(numImages)+'labels.tsv'
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
    visualize_fully_connected_embedding.create_labels_file(video.embeddings_folder)
    visualize_fully_connected_embedding.visualize(video.embeddings_folder)
