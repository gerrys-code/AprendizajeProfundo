import os
import tensorflow as tf


# silences Tensorflow boots logs

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def run():
    #Define the graph
    #input variables
    x = tf.placeholder(tf.int32,name="x")
    y = tf.placeholder(tf.int32,name="y")
    z = tf.placeholder(tf.int32,name="z")
    #Define the operations
    op1 = tf.add(x,y,name="op1")
    op2 = tf.pow(op1,z,name="PowOp")

    #Evaluating the graph

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('graphs/my_placeholders', sess.graph)
        #dictionary to assign values
        feed_dict = {x:1 , y:2, z:3}
        fetches = [op1,op2]
        print (sess.run(fetches,feed_dict))


def main(args):
    #run()
    print (args)
    print ("Here")
    return 0


if __name__ == '__main__':
  import sys
  sys.exit(main(sys.argv))
