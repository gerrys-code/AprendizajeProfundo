""" Simple convolutional neural network.
UNAM IIMAS
Course:     Deep Learning
Professor:  Gibran Fuentes Pineda
Assistant:  Berenice Montalvo Lezama
"""

import os

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Logreg:

  def __init__(self):
    """ Creates the model """
    self.def_input()
    self.def_params()
    self.def_model()
    self.def_output()
    self.def_loss()
    self.def_metrics()
    self.add_summaries()

  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input'):
      # placeholder for X
      self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
      # placeholder for Y
      self.Y_true = tf.placeholder(tf.float32, [None, 10], name='Y')
      # flattens X -> XR : 28x28 -> 784
      self.XR = tf.reshape(self.X, [-1, 784], name='XR')

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):
      # First convolutional layer - maps one grayscale image to 32 feature maps.
      with tf.name_scope('conv1'):
        self.W_cn1 = self.weight_variable([5, 5, 1, 32])
        self.b_cn1 = self.bias_variable([32])

      # Second convolutional layer -- maps 32 feature maps to 64.
      with tf.name_scope('conv2'):
        self.W_cn2 = self.weight_variable([5, 5, 32, 64])
        self.b_cn2 = self.bias_variable([64])

      # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
      # is down to 7x7x64 feature maps -- maps this to 1024 features.
      with tf.name_scope('fc1'):
        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])

      # Map the 1024 features to 10 classes, one for each digit
      with tf.name_scope('fc2'):
        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])

  def def_model(self):
    """ Defines the model """
    W_cn1 = self.W_cn1
    b_cn1 = self.b_cn1
    W_cn2 = self.W_cn2
    b_cn2 = self.b_cn2
    W_fc1 = self.W_fc1
    b_fc1 = self.b_fc1
    W_fc2 = self.W_fc2
    b_fc2 = self.b_fc2
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
      h_cn1 = tf.nn.relu(self.conv2d(self.X, W_cn1) + b_cn1)
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
      h_pool1 = self.max_pool_2x2(h_cn1)
    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
      h_cn2 = tf.nn.relu(self.conv2d(h_pool1, W_cn2) + b_cn2)
    # Second pooling layer.
    with tf.name_scope('pool2'):
      h_pool2 = self.max_pool_2x2(h_cn2)
    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
      self.Y_logt = tf.matmul(h_fc1, W_fc2) + b_fc2
      self.Y_pred = tf.nn.softmax(self.Y_logt)

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y_true, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):
      # cross entropy
      self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y_true, logits=self.Y_logt)
      self.loss = tf.reduce_mean(self.cross_entropy)

  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics'):
      cmp_labels = tf.equal(self.label_true, self.label_pred)
      self.accuracy = tf.reduce_mean(tf.cast(cmp_labels, tf.float32), name='accuracy')

  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('accuracy', self.accuracy)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  def weight_variable(self, shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def train(self, data):
    """ Trains the model """
    # creates optimizer
    grad = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    # setup minimize function
    optimizer = grad.minimize(self.loss)

    # opens session
    with tf.Session() as sess:
      # initialize variables (params)
      sess.run(tf.global_variables_initializer())
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/ex7_cnn_train')
      test_writer = tf.summary.FileWriter('graphs/ex7_cnn_test')
      train_writer.add_graph(sess.graph)

      #  batches
      X_test, Y_test = data.test.images, data.test.labels
      print (X_test.shape)
      print ("===========Here================")
      print (Y_test.shape)

      # training loop
      for i in range(500):
        print (i)
        # train batch
        X_train, Y_train = data.train.next_batch(100)
        print ("====================Train===============")
        print ((X_train.shape))
        print (Y_train.shape)
        break

        # evaluation with train data
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
        #~ fetches = [self.loss]
        #~ train_loss = sess.run(fetches, feed_dict=feed_dict)
        fetches = [self.loss, self.accuracy, self.summary]
        train_loss, train_acc, train_summary = sess.run(fetches, feed_dict=feed_dict)
        train_writer.add_summary(train_summary, i)

        # evaluation with test data
        feed_dict = {self.X: X_test, self.Y_true: Y_test}
        fetches = [self.loss, self.accuracy, self.summary]
        test_loss, test_acc, test_summary = sess.run(fetches, feed_dict=feed_dict)
        test_writer.add_summary(test_summary, i)

        # train step
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
        fetches = [optimizer]
        sess.run(fetches, feed_dict=feed_dict)


        # console output
        msg = "I{:3d} loss: ({:6.2f}, {:6.2f}), acc: ({:6.2f}, {:6.2f})"
        msg = msg.format(i, train_loss, test_loss, train_acc, test_acc)
        print(msg)


def run():
  # Tensorflow integrates MNIST dataset
  mnist = read_data_sets('data', one_hot=True, reshape=False, validation_size=0)
  # defines our model
  model = Logreg()
  # trains our model
  model.train(mnist)


def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
