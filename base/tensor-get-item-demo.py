# -*- coding: utf-8 -*-
import tensorflow as tf

foo1 = tf.constant([1,2,3,4,5,6])
foo2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    # strip leading and tailing 2 elememts
    print(sess.run(foo1[2:-2]))

    # skip row and revers ever column
    print(foo2[::2, ::-1].eval(session = sess))

    # Insert another dimension
    print(foo2[tf.newaxis, :, :].eval(session = sess))
    print(foo2[:, tf.newaxis, :].eval(session = sess))
    print(foo2[:, :, tf.newaxis].eval(session = sess))

    # Ellipses (3 equivalent operations)
    print(foo2[tf.newaxis, :, :].eval(session = sess))
    print(foo2[tf.newaxis, ...].eval(session = sess))
    print(sess.run(foo2[tf.newaxis]))