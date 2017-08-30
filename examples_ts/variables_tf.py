import tensorflow as tf

a = tf.Variable(2,name="scalar")
b = tf.Variable([2,3],name="vector")

#create a variable c as a 2x2 matrix

c = tf.Variable([[0,1],[2,3]],name="matrix")

#create variable w as 784x10 tensor, filed with zeros

w = tf.Variable(tf.zeros([784,10]))
