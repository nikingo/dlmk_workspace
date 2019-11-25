import math
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
import argparse
import cv2
from glob import glob
import os

from prepare import data_load
from prepare import get_shuffled_batch_ind

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

num_classes = 2
img_height, img_width = 64, 64

base_path = os.path.abspath(os.path.dirname(__file__))
data_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\train\\images"
test_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\test\\images"
data_path = os.path.join(base_path, data_dir)
test_path = os.path.join(base_path, test_dir)

def conv2d(x, k=3, in_num=1, out_num=32, strides=1, activ=None, bias=True, name='conv'):
    w = tf.Variable(tf.random_normal([k, k, in_num, out_num]), name=name+'_w')
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.nn.bias_add(x, b)
    if activ is not None:
        x = activ(x)
    return x


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def fc(x, in_num=100, out_num=100, bias=True, activ=None, name='fc'):
    w = tf.Variable(tf.random_normal([in_num, out_num]), name=name+'_w')
    x = tf.matmul(x, w)
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.add(x, b)
    if activ is not None:
        x = activ(x)
    return x

def Mynet(x, keep_prob):
    x = conv2d(x, k=3, in_num=3, out_num=32, activ=tf.nn.relu, name='conv1_1')
    x = conv2d(x, k=3, in_num=32, out_num=32, activ=tf.nn.relu, name='conv1_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=32, out_num=64, activ=tf.nn.relu, name='conv2_1')
    x = conv2d(x, k=3, in_num=64, out_num=64, activ=tf.nn.relu, name='conv2_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=64, out_num=128, activ=tf.nn.relu, name='conv3_1')
    x = conv2d(x, k=3, in_num=128, out_num=128, activ=tf.nn.relu, name='conv3_2')
    x = maxpool2d(x, k=2)

    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    x = fc(x, in_num=w*h*c, out_num=1024, activ=tf.nn.relu, name='fc1')
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)
    x = fc(x, in_num=1024, out_num=num_classes, name='fc_out')
    return x


def train_net(train, accuracy, loss, xs, ys):
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        ind_batch = get_shuffled_batch_ind(len(ys), 16, 10)
        iter_per_epoch = len(ys) // 32

        running_loss = 0
        for i, (batch_xs, batch_ys) in enumerate(zip(xs[ind_batch], ys[ind_batch])):

            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})
            
            print("iter >>", i+1, ',loss >>', los / 32, ',accuracy >>', acc)

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(base_path, 'tf_cnn.ckpt'))


def test_net(xs, ys):

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(base_path, "tf_cnn.ckpt"))

        for i in range(len(ys)):

                input_shape = xs[i].shape
                #print(input_shape)
                x = xs[i].reshape(1, input_shape[0], input_shape[1],input_shape[2])
                y = ys[i]

                pred = sess.run([out], feed_dict={X:x, keep_prob:1.})[0]
                print('label:', y, 'out:', pred)


#train
X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# 自分でlayerを定義した時
logits = Mynet(X, keep_prob)

preds = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

xs, ys = data_load(data_path, 64, 64, hflip=True, vflip=True, rot=[angle for angle in range(0,360,10)])
xs = xs.transpose(0, 2, 3, 1)
ys = np.identity(num_classes)[ys]
print(ys)

train_net(train, accuracy, loss, xs, ys)


#test
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

#logits = Mynet(X, train=False)
logits = Mynet(X, keep_prob)
out = tf.nn.softmax(logits)

xs, ys = data_load(test_path, 64, 64)
xs = xs.transpose(0, 2, 3, 1)
ys = np.identity(num_classes)[ys]
print(ys)

test_net(xs, ys)