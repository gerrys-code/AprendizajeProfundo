import tensorflow as tf

a = tf.constant(2,name="a")
b = tf.constant(3,name="b")

x = tf.add(a,b)

#writer = tf.summary.FileWriter('./graphs',sess.graph)
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print (sess.run(x))

writer.close()
