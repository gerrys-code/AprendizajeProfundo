""" Simple neural network.
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
      L0, L1, OP = 784, 100, 10
      # L0
      self.W0 = tf.Variable(tf.truncated_normal([L0, L1], stddev=0.1), name='W0')
      self.B0 = tf.Variable(tf.truncated_normal([L1], stddev=0.1), name='B0')
      # L1
      self.W1 = tf.Variable(tf.truncated_normal([L1, OP], stddev=0.1), name='W1')
      self.B1 = tf.Variable(tf.truncated_normal([OP], stddev=0.1), name='B1')

  def def_model(self):
    """ Defines the model """
    X, W0, B0, W1, B1 = self.XR, self.W0, self.B0, self.W1, self.B1
    # model
    with tf.name_scope('model'):
      Y0 = tf.nn.sigmoid(tf.matmul(X, W0) + B0)
      Y1 = tf.nn.softmax(tf.matmul(Y0, W1) + B1)
      self.Y_pred = Y1

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y_true, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):
      # cross entropy
      self.loss = - tf.reduce_mean(self.Y_true * tf.log(self.Y_pred), name='loss') * 1000

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
      tf.summary.histogram('W0', self.W0)
      tf.summary.histogram('B0', self.B0)
      tf.summary.histogram('W1', self.W1)
      tf.summary.histogram('B1', self.B1)
      # groups summaries
      self.summary = tf.summary.merge_all()

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
      train_writer = tf.summary.FileWriter('graphs/ex6_neuralnet_train')
      test_writer = tf.summary.FileWriter('graphs/ex6_neuralnet_test')
      train_writer.add_graph(sess.graph)

      #  batches
      X_test, Y_test = data.test.images, data.test.labels

      # training loop
      for i in range(550*3):

        # train batch
        X_train, Y_train = data.train.next_batch(100)

        # evaluation with train data
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
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
