import os
import glob
import numpy as np

def forward(xs, w, b):
    return np.dot(xs, w) + b

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def XOR_forward(xs, w, b, w_out, b_out):
    z = sigmoid(forward(xs, w, b))
    return z, sigmoid(forward(z, w_out, b_out))

def learn(xs, ts, w, b, w_out, b_out, lr, iteration):

    for _ in range(iteration):

        #print("w s:", w)

        z, ys = XOR_forward(xs, w, b, w_out, b_out)
        #print("z", z)
        #print("ys:", ys)

        En_out = -(ts - ys) * ys * (1 - ys) #二乗誤差( 1/2(ts-ys) ) の ysに関する微分
        #print("En:", En_out)
        En_w = np.dot(z.T, En_out)         #二乗誤差( 1/2(ts-ys) )のw_outに関する微分 = dys/dw_out * dE/dys
        #print("En_w:", En_w)
        En_b = np.dot(np.ones(ts.shape[0]), En_out)      #二乗誤差のbに関する微分 = dys/db_out * dE/dys
        #print("En_b:", En_b)
        En_z = np.dot(w_out, En_out.T)  #二乗誤差のzに関する微分 = dys/dz * dE/dys
        w_out -= En_w * lr      #パラメータの更新
        b_out -= En_b * lr
        #print("w_out", "b_out", w_out, b_out)

        #print("En_z:", En_z)

        En_y = (1 - z) * z * En_z.T       #二乗誤差のy=[y1,y2]に関する微分 = dz / dy * dE/dz  ※(z = sigmoid(y))

        #print("En_y:", En_y)

        En_w_1 = np.dot(xs.T, En_y)     #二乗誤差のwに関する微分 = dy / dw * dE / dy
        #print("En_w_1:", En_w_1)
        w -= En_w_1 * lr

        En_b_1 = np.dot(np.ones(ts.shape[0]), En_y)     #二乗誤差のbに関する微分 = dy / db * dE / dy
        #print("En_b_1:", En_b_1)
        b -= En_b_1 * lr                #パラメータの更新
        #print("w, b", w, b)

        #print("w e:", w, "b e", b)

def XOR_learn():

        np.random.seed(0)
        xs = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
        ts = np.array([[0], [1], [1], [0]], dtype=np.float32)

        w = np.random.normal(0, 1, [2,2])
        b = np.random.normal(0, 1, [2])
        w_out = np.random.normal(0, 1, [2, 1])
        b_out = np.random.normal(0, 1, [1])
        print("w s:", w, "b s:", b)
        print("w_out s:", w_out, "b_out s:", b_out)

        #iteration = 1
        iteration = 5000

        lr = 0.1

        learn(xs, ts, w, b, w_out, b_out, lr, iteration)

        print(XOR_forward(xs, w, b, w_out, b_out))

XOR_learn()