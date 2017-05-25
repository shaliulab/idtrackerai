import cPickle as pickle
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = './ckpt'
PATH_TO_SPRITE_IMAGE = './4_fish_1024imgs_sprite.png'
PATH_TO_LABELS = './4_fish_1024labels.tsv'
PATH_TO_EMBEDDING_IMAGES = './4_fish_1024images.pkl'
PATH_TO_EMBEDDING = './4_fish_1024features.pkl'

emb = pickle.load(open(PATH_TO_EMBEDDING, "rb"))
emb2 = pickle.load(open(PATH_TO_EMBEDDING_IMAGES, "rb"))

embedding_var = tf.Variable(emb, name='features')
embedding_var_images = tf.Variable(emb2, name='images')

step = tf.Variable(0, name='step', trainable=False)
saver = tf.train.Saver()

with tf.Session() as session:
    tf.global_variables_initializer().run()
    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding2 = config.embeddings.add()
    embedding2.tensor_name = embedding_var_images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path =  PATH_TO_LABELS
    embedding2.metadata_path =  PATH_TO_LABELS

    embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
    embedding2.sprite.image_path = PATH_TO_SPRITE_IMAGE
    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([32, 32])
    embedding2.sprite.single_image_dim.extend([32, 32])

    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(summary_writer, config)

    step.assign(0).eval()

    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
