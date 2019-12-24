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

#tf layer をラッパーして書くやりかた

num_classes = 2
img_height, img_width = 64, 64

base_path = os.path.abspath(os.path.dirname(__file__))
data_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\train\\images"
test_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\test\\images"
data_path = os.path.join(base_path, data_dir)
test_path = os.path.join(base_path, test_dir)

# TensorBoard情報出力ディレクトリ
log_dir = os.path.join(base_path, 'log_data')
print(log_dir)


# convolution 層のラッパー,
def conv2d(x, k=3, in_num=1, out_num=32, strides=1, activ=None, bias=True, name='conv'):
    w = tf.Variable(tf.random_normal([k, k, in_num, out_num]), name=name+'_w')  #重みの初期値の変数を定義
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')    #畳み込み演算 stridesの1つ目と4つ目はN,Cなので当然1、SAMEは、入力と出力のサイズを合わせるようなパディング
    tf.add_to_collections('vars', w)    #コレクションに変数を名前付きで保存
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.nn.bias_add(x, b)    #バイアスとして足し算処理、複数チャンネルにも対応するようブロードキャストして加算する
    if activ is not None:
        x = activ(x)    #引数の関数オブジェクトでアクティベーションする
    
    tf.summary.histogram('conv_weights', w)
    tf.summary.histogram('conv_bias', b)
    return x


def maxpool2d(x, k=2):  #特に意味のないラッパー
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# 全結合層のラッパー
def fc(x, in_num=100, out_num=100, bias=True, activ=None, name='fc'):
    w = tf.Variable(tf.random_normal([in_num, out_num]), name=name+'_w')
    x = tf.matmul(x, w) #行列積
    tf.add_to_collections('vars', w)
    if bias:
        b = tf.Variable(tf.random_normal([out_num]), name=name+'_b')
        tf.add_to_collections('vars', b)
        x = tf.add(x, b)    #1-Dの加算
    if activ is not None:
        x = activ(x)
    
    return x

def Mynet(x, keep_prob):
    #定義したラッパーを使って計算グラフを作成
    x = conv2d(x, k=3, in_num=3, out_num=32, activ=tf.nn.relu, name='conv1_1')
    x = conv2d(x, k=3, in_num=32, out_num=32, activ=tf.nn.relu, name='conv1_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=32, out_num=64, activ=tf.nn.relu, name='conv2_1')
    x = conv2d(x, k=3, in_num=64, out_num=64, activ=tf.nn.relu, name='conv2_2')
    x = maxpool2d(x, k=2)
    x = conv2d(x, k=3, in_num=64, out_num=128, activ=tf.nn.relu, name='conv3_1')
    x = conv2d(x, k=3, in_num=128, out_num=128, activ=tf.nn.relu, name='conv3_2')
    x = maxpool2d(x, k=2)

    mb, h, w, c = x.get_shape().as_list()   #ネットワークの返り値のshapeを取得
    x = tf.reshape(x, [-1, h*w*c])  #画像・チャンネルを一列のデータに均す
    x = fc(x, in_num=w*h*c, out_num=1024, activ=tf.nn.relu, name='fc1')
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob)   #ドロップアウト(一定割合のニューロンの出力が0)
    x = fc(x, in_num=1024, out_num=num_classes, name='fc_out')
    return x


def train_net(train, accuracy, loss, merged, xs, ys):
    config = tf.ConfigProto()
    sess = tf.InteractiveSession(config=config)

    sess.run(tf.global_variables_initializer()) #おまじない。グローバルに定義されたvariableをvariable_initializerに渡す

    ind_batch = get_shuffled_batch_ind(len(ys), 32, 5)
    iter_per_epoch = len(ys) // 32

    running_loss = 0

    # SummaryWriterでグラフを書く(これより後のコマンドはグラフに出力されない)
    summary_writer = tf.summary.FileWriter(log_dir , sess.graph)

    for i, (batch_xs, batch_ys) in enumerate(zip(xs[ind_batch], ys[ind_batch])):

        _, sammary = sess.run([train, merged], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.5})   #プレースホルダーの中身を確定、計算グラフ実行(train, accuracy, lossを実行)
        
        #print("iter >>", i+1, ',loss >>', los / 32, ',accuracy >>', acc)

        summary_writer.add_summary(sammary, i)

    saver = tf.train.Saver()    #学習結果の保存準備
    saver.save(sess, os.path.join(base_path, 'tf_cnn.ckpt'))    #セッション結果を保存
    
    summary_writer.close()


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
X = tf.placeholder(tf.float32, [None, img_height, img_width, 3])    #入力のプレースホルダーを定義(noneはなんでもいい値)
Y = tf.placeholder(tf.float32, [None, num_classes]) #ラベルのプレースホルダーを定義
keep_prob = tf.placeholder(tf.float32)  #ドロップアウトの確率のプレースホルダー

# 自分でlayerを定義した時
logits = Mynet(X, keep_prob)

preds = tf.nn.softmax(logits)   #Mynetの結果をsoftmaxする計算の定義
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=Y))  #Mynetの結果とラベルとの間で誤差関数の計算をする計算の定義
optimizer = tf.train.AdamOptimizer(learning_rate=0.001) #最適化手段の定義
train = optimizer.minimize(loss)    #定義した最適化で、lossの結果に対して最適化を行う計算の定義

correct_pred = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))   #
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    #

# SummaryWriterでグラフを書く
tf.summary.image('input', X, 10)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

xs, ys = data_load(data_path, 64, 64, hflip=True, vflip=True, rot=[angle for angle in range(0,360,10)])
xs = xs.transpose(0, 2, 3, 1)
ys = np.identity(num_classes)[ys]   #one-hot化
print(ys)

train_net(train, accuracy, loss, merged, xs, ys)    #定義した計算を渡して学習sessionを実行させる


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