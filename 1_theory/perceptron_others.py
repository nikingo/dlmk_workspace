import os
import glob
import numpy as np

def forward(xs, w, b):
    return np.dot(xs, w) + b

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def learn(xs, ts, w, b, lr, iteration):

    for _ in range(iteration):

        #print("w s:", w)

        ys = sigmoid(forward(xs, w, b))

        #print("ys:", ys)

        En = -(ts - ys) * ys * (1 - ys) #sigmoid(ys)のysに関する微分
        #print("En:", En)
        En_w = np.dot(xs.T, En)         #sigmoid(xs.w + b)のwに関する微分
        En_b = np.dot(np.ones((xs.shape[0],)), En)      #sigmoid(xs.w + b)のbに関する微分
        w -= En_w * lr
        b -= En_b * lr

        print("w e:", w, "b e", b)

def OR_learn():
        xs = np.array([[0,0], [0,1], [1,0], [1,1]])
        ts = np.array([0, 1, 1, 1])

        np.random.seed(0)
        w = np.random.normal(0., 1, (2))
        b = np.random.normal(0., 1, (1))
        print("w s:", w, "b s:", b)

        iteration = 5000

        lr = 0.1

        learn(xs, ts, w, b, lr, iteration)

        print(forward(xs, w, b))
        print(sigmoid(forward(xs, w, b)))

def NOT_learn():
        xs = np.array([[0], [1]])
        ts = np.array([1, 0])

        np.random.seed(0)
        w = np.random.normal(0., 1, (1))
        b = np.random.normal(0., 1, (1))
        print("w s:", w, "b s:", b)

        iteration = 5000

        lr = 0.1

        learn(xs, ts, w, b, lr, iteration)

        print(forward(xs, w, b))
        print(sigmoid(forward(xs, w, b)))

def XOR_learn():
        xs = np.array([[0,0], [0,1], [1,0], [1,1]])
        ts = np.array([0, 1, 1, 0])

        np.random.seed(0)
        w = np.random.normal(0., 1, (2))
        b = np.random.normal(0., 1, (1))
        print("w s:", w, "b s:", b)

        iteration = 5000

        lr = 0.1

        learn(xs, ts, w, b, lr, iteration)

        print(forward(xs, w, b))
        print(sigmoid(forward(xs, w, b)))

#OR_learn()
#NOT_learn()
#XOR_learn()