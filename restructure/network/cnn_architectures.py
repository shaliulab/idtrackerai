import tensorflow as tf

def cnn_model_1(x_pl):

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
        maxpool1, 64, 5, 1,
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
        maxpool2, 100, 5, 1,
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

    return y_logits
