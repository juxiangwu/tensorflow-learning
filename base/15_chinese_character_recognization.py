# -*- coding: utf-8 -*-
"""
Tensorflow 汉字识别
from:http://blog.csdn.net/u014365862/article/details/53869837
"""
import os
import numpy as np
import struct
import PIL.Image

train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"


# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


import scipy.misc
from sklearn.utils import shuffle
import tensorflow as tf

# 我取常用的前140个汉字进行测试
char_set = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"


def resize_and_normalize_image(img):
    # 补方
    pad_size = abs(img.shape[0] - img.shape[1]) // 2
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # 缩放
    img = scipy.misc.imresize(img, (64 - 4 * 2, 64 - 4 * 2))
    img = np.lib.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=255)
    assert img.shape == (64, 64)

    img = img.flatten()
    # 像素值范围-1到1
    img = (img - 128) / 128
    return img


# one hot
def convert_to_one_hot(char):
    vector = np.zeros(len(char_set))
    vector[char_set.index(char)] = 1
    return vector


# 由于数据量不大, 可一次全部加载到RAM
train_data_x = []
train_data_y = []

for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    if tagcode_unicode in char_set:
        train_data_x.append(resize_and_normalize_image(image))
        train_data_y.append(convert_to_one_hot(tagcode_unicode))

# shuffle样本
train_data_x, train_data_y = shuffle(train_data_x, train_data_y, random_state=0)

batch_size = 128
num_batch = len(train_data_x) // batch_size

text_data_x = []
text_data_y = []
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    if tagcode_unicode in char_set:
        text_data_x.append(resize_and_normalize_image(image))
        text_data_y.append(convert_to_one_hot(tagcode_unicode))
# shuffle样本
text_data_x, text_data_y = shuffle(text_data_x, text_data_y, random_state=0)

X = tf.placeholder(tf.float32, [None, 64 * 64])
Y = tf.placeholder(tf.float32, [None, 140])
keep_prob = tf.placeholder(tf.float32)


def chinese_hand_write_cnn():
    x = tf.reshape(X, shape=[-1, 64, 64, 1])
    # 3 conv layers
    w_c1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
    b_c1 = tf.Variable(tf.zeros([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c2 = tf.Variable(tf.zeros([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """
    # 训练开始之后我就去睡觉了, 早晨起来一看, 白跑了, 准确率不足10%; 把网络变量改少了再来一发
    w_c3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    b_c3 = tf.Variable(tf.zeros([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    """

    # fully connect layer
    w_d = tf.Variable(tf.random_normal([8 * 32 * 64, 1024], stddev=0.01))
    b_d = tf.Variable(tf.zeros([1024]))
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(tf.random_normal([1024, 140], stddev=0.01))
    b_out = tf.Variable(tf.zeros([140]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


def train_hand_write_cnn():
    output = chinese_hand_write_cnn()

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    # TensorBoard
    tf.scalar_summary("loss", loss)
    tf.scalar_summary("accuracy", accuracy)
    merged_summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 命令行执行 tensorboard --logdir=./log  打开浏览器访问http://0.0.0.0:6006
        summary_writer = tf.train.SummaryWriter('./log', graph=tf.get_default_graph())

        for e in range(50):
            for i in range(num_batch):
                batch_x = train_data_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_data_y[i * batch_size: (i + 1) * batch_size]
                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
                # 每次迭代都保存日志
                summary_writer.add_summary(summary, e * num_batch + i)
                print(e * num_batch + i, loss_)

                if e * num_batch + i % 100 == 0:
                    # 计算准确率
                    acc = accuracy.eval({X: text_data_x[:500], Y: text_data_y[:500], keep_prob: 1.})
                    # acc = sess.run(accuracy, feed_dict={X: text_data_x[:500], Y: text_data_y[:500], keep_prob: 1.})
                    print(e * num_batch + i, acc)


train_hand_write_cnn()