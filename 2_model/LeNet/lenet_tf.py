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

#ToDo:ディレクトリ違ってもimportしたい
base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(base_path, '..\\'))
from prepare import data_load
from prepare import get_shuffled_batch_ind

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#MNIST 利用のため
import torchvision

#tf layer をラッパーして書くやりかた

num_classes = 10
#img_height, img_width = 32, 32  #ToDo padding で28->32にしたい
img_height, img_width = 28, 28  #ToDo padding で28->32にしたい

data_path = os.path.join(base_path, '..\\mnist\\')
print(data_path)

train_tensor = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_tensor = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=torchvision.transforms.ToTensor())

train_data = train_tensor.data.cpu().numpy()[:10000]
print(type(train_data), train_data.shape)
train_label = train_tensor.targets.cpu().numpy()[:10000]
print(type(train_label), train_label.shape)
test_data = test_tensor.data.cpu().numpy()[:1000]
print(type(test_data), test_data.shape)
test_label = test_tensor.targets.cpu().numpy()[:1000]
print(type(test_label), test_label.shape)


def Mynet(x):

    x = tf.pad(x, tf.constant([[0, 0,], [2, 2], [2, 2], [0, 0]]), "CONSTANT")
    #print(x.shape)
    w = tf.Variable(tf.random_normal([3, 3, 1, 5]), name='w1')  #重みの初期値の変数を定義
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.random_normal([5]), name='b1')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    #print(x.shape)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print(x.shape)

    w = tf.Variable(tf.random_normal([3, 3, 5, 10]), name='w2')  #重みの初期値の変数を定義
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.random_normal([10]), name='b2')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    #print(x.shape)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #print(x.shape)

    mb, h, w, c = x.get_shape().as_list()   #ネットワークの返り値のshapeを取得
    x = tf.reshape(x, [-1, h*w*c])  #画像・チャンネルを一列のデータに均す
    w = tf.Variable(tf.random_normal([w*h*c, num_classes]), name='w3')
    x = tf.matmul(x, w) #行列積
    b = tf.Variable(tf.random_normal([num_classes]), name='b3')
    x = tf.add(x, b)    #1-Dの加算
    #x = tf.nn.relu(x)
    #print(x.shape)

    return x


def train_net(train, preds, accuracy, loss, merged, xs, ys):
    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)

    sess.run(tf.global_variables_initializer()) #おまじない。グローバルに定義されたvariableをvariable_initializerに渡す

    ind_batch = get_shuffled_batch_ind(len(ys), 512, 20)
    iter_per_epoch = len(ys) // 512

    running_loss = 0

    # SummaryWriterでグラフを書く(これより後のコマンドはグラフに出力されない)
    #summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    for i, (batch_xs, batch_ys) in enumerate(zip(xs[ind_batch], ys[ind_batch])):

        #_, sammary = sess.run([train, merged], feed_dict={X: batch_xs, Y: batch_ys})   #プレースホルダーの中身を確定、計算グラフ実行(train, accuracy, lossを実行)
        _, pre, acc, los = sess.run([train, preds, accuracy, loss], feed_dict={X: batch_xs, Y: batch_ys})   #プレースホルダーの中身を確定、計算グラフ実行(train, accuracy, lossを実行)
        print("iter >>", i+1, ',loss >>', los, ',accuracy >>', acc)
        #print(pre)

        #summary_writer.add_summary(sammary, i)

    saver = tf.train.Saver()    #学習結果の保存準備
    saver.save(sess, os.path.join(base_path, 'tf_cnn.ckpt'))    #セッション結果を保存
    
    #summary_writer.close()


#train
X = tf.placeholder(tf.float32, [None, img_height, img_width, 1])    #入力のプレースホルダーを定義(noneはなんでもいい値)
Y = tf.placeholder(tf.float32, [None, num_classes]) #ラベルのプレースホルダーを定義

# 自分でlayerを定義した時
logits = Mynet(X)

preds = tf.nn.softmax(logits)   #Mynetの結果をsoftmaxする計算の定義
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))  #Mynetの結果とラベルとの間で誤差関数の計算をする計算の定義
optimizer = tf.train.AdamOptimizer(learning_rate=0.01) #最適化手段の定義
train = optimizer.minimize(loss)    #定義した最適化で、lossの結果に対して最適化を行う計算の定義

correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))   #
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    #

# SummaryWriterでグラフを書く
tf.summary.image('input', X, 10)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

xs = train_data.reshape(-1, img_height, img_width, 1)
ys = np.identity(num_classes)[train_label]
print(ys)
train_net(train, correct_pred, accuracy, loss, merged, xs, ys)    #定義した計算を渡して学習sessionを実行させる