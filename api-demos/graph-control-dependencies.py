
from __future__ import print_function
import tensorflow as tf

g = tf.get_default_graph()

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    with g.control_dependencies([init]):
        a = tf.placeholder(tf.float32)
        b = tf.placeholder(tf.float32)
        c = a + b
        d = a * b
        e = a / b
        d = a * b
        dval = sess.run(d,feed_dict={a:1,b:2})
        print('dval = ',dval)
    cval = sess.run(c,feed_dict={a:2,b:3})
    print('cval = ',cval)

    #nested control_dependencies
    with g.control_dependencies([a,b,c,d]):
        f = c + d
        fval = sess.run(f,feed_dict={a:2,b:3})
        print('faval = ',fval)
        with g.control_dependencies([f]):
            h = f * c * d
            hval = sess.run(h,feed_dict={a:1,b:2})
            print('hval = ',hval)
