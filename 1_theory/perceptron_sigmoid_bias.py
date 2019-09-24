import os
import glob
import numpy as np

def forward(xs, w, b):
    return np.dot(xs, w) + b

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def learn(xs, ts, w, b, lr):

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

xs = np.array([[0,0], [0,1], [1,0], [1,1]])
#xs = np.hstack([xs, np.ones((xs.shape[0],1))])
ts = np.array([0, 0, 0, 1])

np.random.seed(0)
w = np.random.normal(0., 1, (2))
b = np.random.normal(0., 1, (1))
print("w s:", w, "b s:", b)

iteration = 5000

lr = 0.1

learn(xs, ts, w, b, lr)

print(forward(xs, w, b))
print(sigmoid(forward(xs, w, b)))
