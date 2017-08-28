""" Example to demostrate the assembling and running of a minimal graph.
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

  # creates two constants that are passed to the 'add' operation
  a = tf.add(1, 2)

  # prints only a representation of 'a' in the graph
  print(a)


  ################################################
  # Evaluating the graph

  # builds a new session
  with tf.Session() as sess:

    # runs 'sess' to evaluate 'a' and get its value
   print(sess.run(a))


def main(args):
  run()
  return 0

if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
