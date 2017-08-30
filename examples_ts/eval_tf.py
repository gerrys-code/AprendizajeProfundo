import tensorflow as tf


# W is a random 700x100 variables objects

#W = tf.Variable(tf.truncated_normal([700,10]))
init = tf.global_variables_initializer()
W = tf.Variable(10)
assign_op=W.assign(100)
with tf.Session() as sess:
    sess.run(init)
    sess.run(assign_op)
    print (W.eval())
