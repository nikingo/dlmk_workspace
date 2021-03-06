import os
import glob
import numpy as np

def forward(xs, w):
    return np.dot(xs, w)

def sigmoid(xs):
    return 1.0 / (1.0 + np.exp(-xs))

def learn(xs, ts, w, lr):

    for _ in range(iteration):

        #print("w s:", w)

        ys = sigmoid(forward(xs, w))

        #print("ys:", ys)

        En = -(ts - ys) * ys * (1 - ys) #sigmoid(ys)のysに関する微分
        #print("En:", En)
        En = np.dot(xs.T, En)   #sigmoid(xs.w)のwに関する微分

        w -= En * lr

        print("w e:", w)

xs = np.array([[0,0], [0,1], [1,0], [1,1]])
xs = np.hstack([xs, np.ones((xs.shape[0],1))])
ts = np.array([0, 0, 0, 1])

np.random.seed(0)
w = np.random.normal(0., 1, (3))
print("w s:", w)

iteration = 5000

lr = 0.1

learn(xs, ts, w, lr)

print(forward(xs, w))
print(sigmoid(forward(xs, w)))
