# -*- coding: utf-8 -*-
import tensorflow as tf
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.mul(a,b, name="mul_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")
sess = tf.Session()
output = sess.run(e)
writer = tf.train.SummaryWriter('D:/Develop/DeepLearning/tensorboard', sess.graph)
writer.close()
sess.close()