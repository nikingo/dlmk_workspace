import os
import glob
import cv2
import numpy as np
from dlmk_2_prepare import data_load, get_shuffled_batch_ind
from deep_layer_perceptron_class import Model, AffineLayer, SigmoidLayer, SquareError, SigmoidCrossEntropy

base_path = os.path.abspath(os.getcwd())
data_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\train\\images"
test_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\test\\images"
data_path = os.path.join(base_path, data_dir)
test_path = os.path.join(base_path, test_dir)

img_height, img_width = 64, 64

np.random.seed(0)

xs, ts, file_list = data_load(data_path, img_height, img_width, hflip=True, vflip=True, rot=[angle for angle in range(0,360,10)])
test, testlabels, file_list = data_load(test_path, img_height, img_width)

ind_batch = get_shuffled_batch_ind(len(ts), 64, 200)
#print(ts[ind_batch])
print(xs.shape)
print(np.array(ind_batch).shape)

xs = xs.reshape(xs.shape[0], -1)   #画像を一次元ベクトル化
print(xs.shape)

test = test.reshape(test.shape[0], -1)
print(test.shape)
print(testlabels.shape)

def sample1():

    l1 = AffineLayer(xs.shape[1], 256)
    l1a = SigmoidLayer()
    l2 = AffineLayer(256, 32)
    l2a = SigmoidLayer()
    l3 = AffineLayer(32, 1)
    l3a = SigmoidLayer()
    #layers = [l1, l1a, l2, l2a, l3, l3a]
    layers = [l1, l1a, l2, l2a, l3, l3a]

    loss_layer = SquareError()

    model = Model(layers, loss_layer, lr=0.01)

    print("test before learning")
    for test_xs, test_ts in zip(test, testlabels):
        test_ys = model.predict(test_xs, test_xs)
        print(test_ys, test_ts)

    print("start leaning")
    for batch_xs, batch_ts in zip(xs[ind_batch], ts[ind_batch]):
    #for i in range(1):
        #batch_xs = xs[0:1]
        #batch_ts = ts[0:1].reshape(-1, 1)
        
        #print(batch_xs.shape, batch_xs)
        #print(batch_ts.shape, batch_ts)

        ys = model.forward(batch_xs, batch_ts)
        model.backward(ys, batch_ts.reshape(-1, 1))
        #print(En)

        #print(model.layers[0].w[0])
        print(model.loss(batch_xs, batch_ts.reshape(-1, 1)))
        #print(ys, batch_ts.reshape(-1, 1))

    print("end leaning")    # ちゃんと誤差は低下するが，学習しきれない

    print("test after learning")
    for test_xs, test_ts in zip(test, testlabels):
        test_ys = model.predict(test_xs, test_ts)
        print(test_ys, test_ts)


def sample2():

    l1 = AffineLayer(xs.shape[1], 256)
    l1a = SigmoidLayer()
    l2 = AffineLayer(256, 32)
    l2a = SigmoidLayer()
    l3 = AffineLayer(32, 1)
    #layers = [l1, l1a, l2, l2a, l3, l3a]
    layers = [l1, l1a, l2, l2a, l3]

    loss_layer = SigmoidCrossEntropy()

    model = Model(layers, loss_layer, lr=0.001)

    print("test before learning")
    for test_xs, test_ts in zip(test, testlabels):
        test_ys = model.predict(test_xs, test_ts)
        print(test_ys, test_ts)

    print("start leaning")
    for batch_xs, batch_ts in zip(xs[ind_batch], ts[ind_batch]):
    #for i in range(1):
        #batch_xs = xs[0:1]
        #batch_ts = ts[0:1].reshape(-1, 1)
        
        #print(batch_xs.shape, batch_xs)
        #print(batch_ts.shape, batch_ts)

        ys = model.forward(batch_xs, batch_ts)
        model.backward(ys, batch_ts.reshape(-1, 1))
        #print(En)

        #print(model.layers[0].w[0])
        print(model.loss(batch_xs, batch_ts.reshape(-1, 1)))
        #print(ys, batch_ts.reshape(-1, 1))

    print("end leaning")    # ちゃんと誤差は低下するが，学習しきれない

    print("test after learning")
    for test_xs, test_ts in zip(test, testlabels):
        test_ys = model.loss_layer.sigmoid(model.predict(test_xs, test_ts))
        print(test_ys, test_ts)


#sample1()
sample2()

# for img, label in zip(xs[ind_batch], ys[ind_batch]):
#     img = (img[0] * 255).astype(np.uint8).transpose(1,2,0)
#     print(label[0])
#     cv2.imshow('aa', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# for img, label in zip(test, testlabels):
#     img = (img * 255).astype(np.uint8).reshape(3, 64, 64).transpose(1,2,0)
#     print(label)
#     cv2.imshow('aa', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
