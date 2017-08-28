""" Simple logistic regresion.
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
      # perceptrons weights initialized with zeros
      self.W = tf.Variable(tf.zeros([784, 10]), name='W')
      # bias weights initialized with zeros
      self.B = tf.Variable(tf.zeros([10]), name='B')

  def def_model(self):
    """ Defines the model """
    X, W, B = self.XR, self.W, self.B
    # model
    with tf.name_scope('model'):
      self.Y_pred = tf.nn.softmax(tf.matmul(X, W) + B)

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      # gets the index of the greater element for each row (example)
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
      tf.summary.histogram('W', self.W)
      tf.summary.histogram('B', self.B)
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
      train_writer = tf.summary.FileWriter('graphs/ex5_logreg_train')
      test_writer = tf.summary.FileWriter('graphs/ex5_logreg_test')
      train_writer.add_graph(sess.graph)

      #  batches
      X_test, Y_test = data.test.images, data.test.labels

      # training loop
      for i in range(550*1):

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

        # console output
        msg = "I{:3d} loss: ({:6.2f}, {:6.2f}), acc: ({:6.2f}, {:6.2f})"
        msg = msg.format(i, train_loss, test_loss, train_acc, test_acc)
        print(msg)

        # train step
        feed_dict = {self.X: X_train, self.Y_true: Y_train}
        fetches = [optimizer]
        sess.run(fetches, feed_dict=feed_dict)


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
