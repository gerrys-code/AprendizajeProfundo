import tensorflow as tf

# Create a session, assign it to variables sess so we can call it later.

a = tf.add(2,3)

print (a)


#sess = tf.Session()
#print (sess.run(a))
#sess.close()

""" A Session object encapsulates the enviroment in which
Operations objects are exacuted, and Tensor objects are
evaluated.
"""
with tf.Session() as sess:
    print (sess.run(a))
