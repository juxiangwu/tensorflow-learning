from __future__ import print_function
import tensorflow as tf

# Creates a scope called "nested"
with tf.Graph().as_default() as g:
    c = tf.constant(5.0, name="c")
    assert c.op.name == "c"
    c_1 = tf.constant(6.0, name="c")
    assert c_1.op.name == "c_1"

    with g.name_scope("nested") as scope:
      nested_c = tf.constant(10.0, name="c")
      assert nested_c.op.name == "nested/c"

# Creates a nested scope called "inner".
with g.name_scope("inner"):
    nested_inner_c = tf.constant(20.0, name="c")
    assert nested_inner_c.op.name == "nested/inner/c"
    with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"
    # Treats `scope` as an absolute name scope, and
    # switches to the "nested/" scope.
    with g.name_scope(""):
        e = tf.constant(50.0, name="e")
        assert e.op.name == "e"
