import os
import glob
import numpy as np

def forward(xs, w, b):
    return np.dot(xs, w) + b

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def XOR_forward(xs, w, b, w2, b2, w_out, b_out):
    z = sigmoid(forward(xs, w, b))
    z2 = sigmoid(forward(z, w2, b2))
    return z, z2, sigmoid(forward(z2, w_out, b_out))

class AffineLayer():

    def __init__(self, input_num, output_num, activation=sigmoid):
        self.w = np.random.normal(0, 1, [input_num, output_num])
        self.b = np.random.normal(0, 1, [output_num])
        self.activation_func = sigmoid
        self.x = None
        self.y = None

    def forward(self, xs):
        self.xs = xs
        self.ys = self.activation_func(np.dot(xs, self.w) + self.b)
        return self.ys

    def backward(self, En):
        En_out = En * self.ys * (1 - self.ys)
        En_w = np.dot(self.xs.T, En_out)         #二乗誤差( 1/2(ts-ys) )のw_outに関する微分 = dys/dw * dE/dys
        En_b = np.dot(np.ones(En_out.shape[0]), En_out)      #二乗誤差のbに関する微分 = dys/db * dE/dys
        En_x = np.dot(En_out, self.w.T) #二乗誤差のxに関する微分 = dys/dx2 * dE/dys

        return En_w, En_b, En_x


def learn(xs, ts, w, b, w2, b2, w_out, b_out, lr, iteration):

    for _ in range(iteration):

        #print("w s:", w)

        z1, z2, ys = XOR_forward(xs, w, b, w2, b2, w_out, b_out)
        #print("z", z)
        #print("z2", z2)
        #print("ys:", ys)

        En_out = -(ts - ys) * ys * (1 - ys) #二乗誤差( 1/2(ts-ys) ) の ysに関する微分
        #print("En:", En_out)
        En_w = np.dot(z2.T, En_out)         #二乗誤差( 1/2(ts-ys) )のw_outに関する微分 = dys/dw_out * dE/dys
        #print("En_w:", En_w)
        En_b = np.dot(np.ones(En_out.shape[0]), En_out)      #二乗誤差のbに関する微分 = dys/db_out * dE/dys
        #print("En_b:", En_b)
        En_z2 = np.dot(En_out, w_out.T) #二乗誤差のz2に関する微分 = dys/dz2 * dE/dys
        w_out -= En_w * lr
        b_out -= En_b * lr
        #print("w_out", "b_out", w_out, b_out)

        #print("En_z:", En_z)

        En_y2 = (1 - z2) * z2 * En_z2       #二乗誤差のy2=[y21,y22]に関する微分 = dz2 / dy2 * dE/dz2  ※(z2 = sigmoid(y2))

        #print("En_y:", En_y)

        En_w_2 = np.dot(z1.T, En_y2)         #二乗誤差のw2に関する微分 = dy2 / dw2 * dE / dy2
        #print("En_w_1:", En_w_1)
        w2 -= En_w_2 * lr

        En_b_2 = np.dot(np.ones(En_y2.shape[0]), En_y2)    #二乗誤差のbに関する微分 = dy2 / db2 * dE / dy
        #print("En_b_1:", En_b_1)
        b2 -= En_b_2 * lr
        #print("w, b", w, b)

        En_z1 = np.dot(En_y2, w2.T) #二乗誤差のzに関する微分 = dy2/dz * dE/dy2


        En_y1 = (1 - z1) * z1 * En_z1   #二乗誤差のy1=[y11,y12]に関する微分 = dz1 / dy1 * dE/dz1  ※(z1 = sigmoid(y1))
        En_w_1 = np.dot(xs.T, En_y1)         #二乗誤差のwに関する微分 = dy1 / dw1 * dE / dy1
        w -= En_w_1 * lr
        En_b_1 = np.dot(np.ones(En_y1.shape[0]), En_y1)    #二乗誤差のbに関する微分 = dy1 / db1 * dE / dy1
        b -= En_b_1 * lr

def XOR_learn():

        np.random.seed(0)
        xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
        ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

        w = np.random.normal(0, 1, [2,2])
        b = np.random.normal(0, 1, [2])
        w2 = np.random.normal(0, 1, [2,2])
        b2 = np.random.normal(0, 1, [2])
        w_out = np.random.normal(0, 1, [2, 1])
        b_out = np.random.normal(0, 1, [1])
        print("w s:", w, "b s:", b)
        print("w_out s:", w_out, "b_out s:", b_out)

        #iteration = 1
        iteration = 10000

        lr = 0.1

        learn(xs, ts, w, b, w2, b2, w_out, b_out, lr, iteration)

        print(XOR_forward(xs, w, b, w2, b2, w_out, b_out))


def learn_layer(xs, ts, layers, iteration, lr=0.1):

    for i in range(iteration):

        y1 = layers[0].forward(xs)
        y2 = layers[1].forward(y1)
        #print(l1.w, l1.b)
        #print(y1)
        #print(l2.w, l2.b)
        #print(y2)

        En = -(ts - y2)

        dw2, db2, dx2 = layers[1].backward(En)
        l2.w -= dw2 * lr
        l2.b -= db2 * lr

        dw, db, dx = layers[0].backward(dx2)
        l1.w -= dw * lr
        l1.b -= db * lr

#XOR_learn()

np.random.seed(0)
xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

l1 = AffineLayer(2, 2)
l2 = AffineLayer(2, 1)
layers = [l1, l2]

learn_layer(xs, ts, layers, 5000)

y1 = layers[0].forward(xs)
y2 = layers[1].forward(y1)

print(y2)