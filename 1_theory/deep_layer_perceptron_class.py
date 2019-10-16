import os
import glob
import numpy as np

class AffineLayer():

    def __init__(self, input_num, output_num):
        self.w = np.random.normal(0, 1, [input_num, output_num])
        self.b = np.random.normal(0, 1, [output_num])

    def forward(self, xs):
        self.xs = xs
        return np.dot(xs, self.w) + self.b

    def backward(self, En, lr):
        En_out = En
        En_w = np.dot(self.xs.T, En_out)         #二乗誤差( 1/2(ts-ys) )のw_outに関する微分 = dys/dw * dE/dys
        En_b = np.dot(np.ones(En_out.shape[0]), En_out)      #二乗誤差のbに関する微分 = dys/db * dE/dys
        En_x = np.dot(En_out, self.w.T) #二乗誤差のxに関する微分 = dys/dx2 * dE/dys

        self.w -= En_w * lr
        self.b -= En_b * lr

        return En_x


class SigmoidLayer():

    def __init__(self):
        self.ys = None

    def sigmoid(self, xs):
        return 1.0 / (1.0 + np.exp(-xs))

    def forward(self, xs):
        self.ys = self.sigmoid(xs)
        return self.ys

    def backward(self, En, lr):
        return En * self.ys * (1 - self.ys)

class SquareError():

    def __init__(self):
        self.ys = None

    def forward(self, ys, ts):
        self.ys = ys
        return ((ts - ys)**2.0) / 2.0

    def backward(self, ys, ts):
        return -(ts - self.ys)

class SigmoidCrossEntropy():

    def __init__(self):
        self.ys = None

    def sigmoid(self, xs):
        return 1.0 / (1.0 + np.exp(-xs))

    def CrossEntropy(self, ys, ts):
        #print(ys)
        return -ts * np.log(ys) - (1.0 - ts) * np.log(1.0 - ys)

    def forward(self, ys, ts):
        ys = self.sigmoid(ys)
        self.ys = ys
        return self.CrossEntropy(ys, ts)

    def backward(self, ys, ts):
        return -(ts - self.ys)


class Model:

    def __init__(self, layers, loss_layer, lr=0.1):
        self.layers = layers
        self.lr = lr
        self.loss_layer = loss_layer

    def forward(self, xs, ts):
        y = xs
        for l in self.layers:
            y = l.forward(y)
        y = self.loss_layer.forward(y, ts)
        return y

    def backward(self, ys, ts):
        dx = self.loss_layer.backward(ys, ts)
        for l in self.layers[::-1]:
            dx = l.backward(dx, self.lr)

    def predict(self, xs, ts):
        y = xs
        for l in self.layers:
            y = l.forward(y)
        return y

    def loss(self, xs, ts):
        l = self.forward(xs, ts)
        return np.mean(l)


def learn_layer(xs, ts, model, iteration):

    for i in range(iteration):

        ys = model.forward(xs, ts)

        #print(ys)

        model.backward(ys, ts)

        print(model.loss(xs, ts))

#XOR_learn()

def XOR_sample():

    np.random.seed(0)
    xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

    l1 = AffineLayer(2, 64)
    l1a = SigmoidLayer()
    l2 = AffineLayer(64, 32)
    l2a = SigmoidLayer()
    l3 = AffineLayer(32, 1)
    l3a = SigmoidLayer()
    layers = [l1, l1a, l2, l2a, l3, l3a]

    loss_layer = SquareError()

    model = Model(layers, loss_layer)

    learn_layer(xs, ts, model, 5000)

    y = model.predict(xs, ts)

    print(y)


def XOR_sample2():

    np.random.seed(0)
    xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

    l1 = AffineLayer(2, 64)
    l1a = SigmoidLayer()
    l2 = AffineLayer(64, 32)
    l2a = SigmoidLayer()
    l3 = AffineLayer(32, 1)
    layers = [l1, l1a, l2, l2a, l3]

    loss_layer = SigmoidCrossEntropy()

    model = Model(layers, loss_layer)

    learn_layer(xs, ts, model, 5000)

    y = model.predict(xs, ts)

    print(y)

#XOR_sample()
#XOR_sample2()