""" Example to demostrate placeholders.
UNAM IIMAS
Course:     Deep Learning
Professor:  Gibran Fuentes Pineda
Assistant:  Berenice Montalvo Lezama
"""

import os

import tensorflow as tf

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def run():

  ################################################
  # Defining the graph

  # input variables
  #
  x = tf.placeholder(tf.int32, name='x')
  y = tf.placeholder(tf.int32, name='y')

  op1 = tf.add(x, y)
  op2 = tf.multiply(x, y)
  op3 = tf.pow(op2, op1, name='PowOp')


  ################################################
  # Evaluating the graph

  with tf.Session() as sess:
    writer = tf.summary.FileWriter('graphs/ex4_placeholders', sess.graph)
    # dictionary to assign values
    feed_dict = {x: 1, y: 2}
    # targets to evaluate
    fetches = [op1, op2, op3]
    print(sess.run(fetches, feed_dict))

    feed_dict = {x: 2, y: 3}
    fetches = [op1, op2, op3]
    print(sess.run(fetches, feed_dict))


def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
