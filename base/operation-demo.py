#-*- coding: utf-8 -*-
import tensorflow as tf

#build a dataflow graph.
c = tf.constant([[1.0,2.0],[5.0,4.0]])
d = tf.constant([[1.0,1.0],[0.0,1.0]])
e = tf.matmul(c,d)
identity = tf.constant([[1.0,0.0],[0.0,1.0]])
with tf.Session() as sess:
    result = sess.run(e)
    print('matrix multiply:')
    print(result)
    f = c + d
    print('matrix add:')
    print(sess.run(f))
    print('matrix sub:')
    g = c - d
    print(sess.run(g))
    h = 1 / c
    print('matrix div:')
    print(sess.run(h))
    res = tf.matmul(identity,c)
    print('identity * c = ')
    print(sess.run(res))
