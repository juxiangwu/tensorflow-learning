# -*- coding: utf-8 -*-
import numpy as np
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')


def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')


def conv_2d(x, w):
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # !!!!!!


def evaluate(y, y_):
    y = tf.arg_max(input=y, dimension=1)
    y_ = tf.arg_max(input=y_, dimension=1)
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32))


def test_cnn(batch_size=50, lr=0.0001, num_iter=20000):
    dataset = input_data.read_data_sets(train_dir='../datas/mnist', one_hot=True)

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='images')  # 后面的卷积操作输入参数必须为‘float32’或者‘float64’
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='labels')

    w_conv1 = weight_variable(shape=[5, 5, 1, 32])
    b_conv1 = bias_variable(shape=[32])
    reshape_x = tf.reshape(x, shape=[-1, 28, 28, 1])  # 省略的形式区别于占位符!!!!!!
    conv1_out = tf.nn.relu(conv_2d(reshape_x, w_conv1) + b_conv1)
    pool1_out = max_pool_2x2(conv1_out)

    w_conv2 = weight_variable(shape=[5, 5, 32, 64])
    b_conv2 = bias_variable(shape=[64])
    conv2_out = tf.nn.relu(conv_2d(pool1_out, w_conv2) + b_conv2)
    pool2_out = max_pool_2x2(conv2_out)

    full_connected_in = tf.reshape(pool2_out, shape=[-1, 7 * 7 * 64])
    w_full_connected = weight_variable(shape=[7 * 7 * 64, 1024])
    b_full_connected = bias_variable(shape=[1024])
    full_connected_out1 = tf.nn.relu(tf.matmul(full_connected_in, w_full_connected) + b_full_connected)
    dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')
    full_connected_out = tf.nn.dropout(x=full_connected_out1, keep_prob=dropout_prob)  # drop out防止过拟合

    w_softmax = weight_variable(shape=[1024, 10])
    b_softmax = bias_variable(shape=[10])
    softmax_in = tf.matmul(full_connected_out, w_softmax) + b_softmax
    softmax_out = tf.nn.softmax(logits=softmax_in, name='softmax_layer')
    Loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_in, labels=y)
    Step_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=Loss)
    accuracy = evaluate(y, softmax_out)  # 在测试数据集上评估算法的准确率
    initialized_variables = tf.initialize_all_variables()

    print('Start to train the convolutional neural network......')
    sess = tf.Session()
    sess.run(fetches=initialized_variables)
    for iter in range(num_iter):
        batch = dataset.train.next_batch(batch_size=batch_size)
        sess.run(fetches=Step_train, feed_dict={x: batch[0], y: batch[1], dropout_prob: 0.5})
        if (iter + 1) % 100 == 0:  # 计算在当前训练块上的准确率
            Accuracy = sess.run(fetches=accuracy, feed_dict={x: batch[0], y: batch[1], dropout_prob: 1})
            print('Iter num %d ,the train accuracy is %.3f' % (iter + 1, Accuracy))

    Accuracy = sess.run(fetches=accuracy, feed_dict={x: dataset.test.images, y: dataset.test.labels, dropout_prob: 1})
    sess.close()
    print('Train process finished, the best accuracy is %.3f' % Accuracy)


if __name__ == '__main__':
    test_cnn()