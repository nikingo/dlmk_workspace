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
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

#MNIST 利用のため
import torchvision

#ToDo:ディレクトリ違ってもimportしたい
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, '..\\'))
from prepare import data_load
from prepare import get_shuffled_batch_ind

num_classes = 10
#img_height, img_width = 32, 32  #ToDo padding で28->32にしたい
img_height, img_width = 28, 28  #ToDo padding で28->32にしたい

data_path = os.path.join(base_path, '..\\mnist\\')
print(data_path)

train_tensor = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_tensor = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data = train_tensor.data.cpu().numpy()[:20000]
print(type(train_data), train_data.shape)
train_label = train_tensor.targets.cpu().numpy()[:20000]
print(type(train_label), train_label.shape)
test_data = test_tensor.data.cpu().numpy()[:2000]
print(type(test_data), test_data.shape)
test_label = test_tensor.targets.cpu().numpy()[:2000]
print(type(test_label), test_label.shape)

def LeNet(x, keep_prob):

    #画像サイズを32にするためのpadding
    x = tf.pad(x, tf.constant([[0, 0,], [2, 2], [2, 2], [0, 0]]), "CONSTANT")
    print(x.shape)

    x = tf.layers.conv2d(inputs=x, filters=6, kernel_size=[5, 5], strides=1, padding='valid', activation=None, name='conv1')
    x = tf.nn.sigmoid(tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2))
    print(x.shape)
    x = tf.layers.conv2d(inputs=x, filters=16, kernel_size=[5, 5], strides=1, padding='valid', activation=None, name='conv2')
    x = tf.nn.sigmoid(tf.layers.max_pooling2d(x, pool_size=[2,2], strides=2))
    print(x.shape)


    mb, h, w, c = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h*w*c])
    print(x.shape)
    x = tf.layers.dense(inputs=x, units=120, activation=None, name='fc1')
    x = tf.layers.dense(inputs=x, units=64, activation=None, name='fc2')
    x = tf.layers.dense(inputs=x, units=num_classes, activation=None, name='fc_out')
    print(x.shape)
    
    return x


# train
def train():
    tf.reset_default_graph()

    # place holder
    X = tf.placeholder(tf.float32, [None, img_height, img_width, 1])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = LeNet(X, keep_prob)
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    train = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    xs = train_data.reshape(-1, img_height, img_width, 1)
    ys = np.identity(num_classes)[train_label]

    ind_batch = get_shuffled_batch_ind(len(ys), 512, 20)
    iter_per_epoch = len(ys) // 512

    running_loss = 0

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        for i, (batch_xs, batch_ys) in enumerate(zip(xs[ind_batch], ys[ind_batch])):

            #_, sammary = sess.run([train, merged], feed_dict={X: batch_xs, Y: batch_ys})   #プレースホルダーの中身を確定、計算グラフ実行(train, accuracy, lossを実行)
            _, acc, los = sess.run([train, accuracy, loss], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})   #プレースホルダーの中身を確定、計算グラフ実行(train, accuracy, lossを実行)
            print("iter >>", i+1, ',loss >>', los, ',accuracy >>', acc)

        saver = tf.train.Saver()    #学習結果の保存準備
        saver.save(sess, os.path.join(base_path, 'tf_cnn.ckpt'))    #セッション結果を保存

# test
def test():
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, img_height, img_width, 1])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    logits = LeNet(X, keep_prob)
    out = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))   #
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    #

    xs = test_data.reshape(-1, img_height, img_width, 1)
    ys = np.identity(num_classes)[test_label]

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(base_path, "tf_cnn.ckpt"))

        acc = sess.run([accuracy], feed_dict={X:xs, Y:ys, keep_prob:1.0})

        print("acc >> {}".format(acc))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")