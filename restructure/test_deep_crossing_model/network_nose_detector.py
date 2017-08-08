import tensorflow as tf
import os
WEIGHT_FALSE_POSITIVES = 50

class ConvNetwork():
    def __init__(self, from_video_path=None):
        self.sesh = tf.Session()
        self.weight_positive = WEIGHT_FALSE_POSITIVES
        handles = self._build_graph()
        print("Building graph....")
        (self.X,self.Y,self.Y_target,self.loss,self.train_step) = handles
        self.saver = tf.train.Saver()

        if from_video_path == None:
            self.sesh.run(tf.global_variables_initializer())
        else:
            self.restore(from_video_path)

    def _build_graph(self):
        X = tf.placeholder(tf.float32,shape=[None,None,None,1])
        conv1 = tf.contrib.layers.conv2d(
            X, 64, 5, 1, activation_fn=tf.nn.relu, padding='SAME', scope="conv1")
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 5, 1, activation_fn=tf.nn.relu, padding='SAME',scope="conv2")
        Y_logit = tf.contrib.layers.conv2d(
            conv2, 1, 3, 1, activation_fn=None, padding='SAME', scope="conv3")
        Y = tf.nn.sigmoid(Y_logit)

        Y_target = tf.placeholder(tf.float32,shape=[None,None,None,1])
        loss = tf.nn.weighted_cross_entropy_with_logits(Y_target, Y_logit, self.weight_positive, name="loss")
        train_step = tf.train.AdamOptimizer().minimize(loss)
        return (X,Y,Y_target,loss,train_step)

    def train(self,batch):
        (batch_images, batch_labels) = batch
        self.train_step.run(session=self.sesh, feed_dict={self.X: batch_images, self.Y_target: batch_labels})

    def prediction(self,images):
        return self.sesh.run(self.Y, feed_dict={self.X: images})

    def save(self, filename, global_step):
        save_path = self.saver.save(self.sesh, filename, global_step=global_step)
        print("Model saved in file:", save_path)

    def restore(self, from_video_path):
        video_folder = os.path.dirname(from_video_path)
        self.nose_detector_ckp_path = os.path.join(video_folder, 'nose_detector_checkpoints')
        ckpt = tf.train.get_checkpoint_state(self.nose_detector_ckp_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring from " + ckpt.model_checkpoint_path)
            self.saver.restore(self.sesh, ckpt.model_checkpoint_path)
        else:
            print("check if any checkpoint has been stored in " + self.nose_detector_ckp_path + ". The folder seems empty, or non existent")
