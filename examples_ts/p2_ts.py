import tensorflow as tf

# Create a session, assign it to variables sess so we can call it later.
x = 2
y = 3

op1 = tf.add(x,y)

op2 = tf.multiply(x,y)

op3 = tf.pow(x,y)

with tf.Session() as sess:
    print (sess.run(op3))
