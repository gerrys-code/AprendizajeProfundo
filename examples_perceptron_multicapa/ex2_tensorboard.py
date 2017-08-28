""" Example to demostrate TensorBoard.
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

  a = tf.add(1, 2)
  print(a)


  ################################################
  # Evaluating the graph

  with tf.Session() as sess:

    # records the graph structure for TensorBoard
    # For visualize TensorBoard you need run the next command 'tensorboard --logdir="./graphs'"
    writer = tf.summary.FileWriter('graphs/ex2_tensorboard', sess.graph)

    print(sess.run(a))


def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
