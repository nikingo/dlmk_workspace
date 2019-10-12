import os
import glob
import numpy as np

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

class AffineLayer():

    def __init__(self, input_num, output_num, activation=sigmoid):
        self.w = np.random.normal(0, 1, [input_num, output_num])
        self.b = np.random.normal(0, 1, [output_num])
        self.activation_func = sigmoid

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

        

class Model:

    def __init__(self, layers, lr=0.1):
        self.layers = layers
        self.lr = lr

    def forward(self, xs):
        y = xs
        for l in self.layers:
            y = l.forward(y)
        
        return y

    def backward(self, En):
        dx = En
        for l in self.layers[::-1]:
            dx = l.backward(dx, self.lr)

    def loss(self, xs, ts):
        ys = self.forward(xs)
        l = np.mean(np.abs(ys - ts))
        return l 


def learn_layer(xs, ts, model, iteration):

    for i in range(iteration):

        y = model.forward(xs)
        
        En = -(ts - y)

        model.backward(En)

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

    model = Model(layers)

    learn_layer(xs, ts, model, 5000)

    y = model.forward(xs)

    print(y)

XOR_sample()