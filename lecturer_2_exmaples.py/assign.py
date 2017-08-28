import tensorflow as tf

my_var = tf.Variable(2,name="my_var")

my_second_var = my_var.assign(2*my_var)


with tf.Session() as sess:
    sess.run(my_var.initializer)
    print (my_var)
    sess.run(my_second_var)
    print (my_second_var.eval())
