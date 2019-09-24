import os
import glob
import numpy as np

def forward(xs, w):
    return np.dot(xs, w)

def learn(xs, ts, w, lr):

    for _ in range(iteration):

        #print("w s:", w)

        ys = forward(xs, w)

        #print("ys:", ys)

        miss = ys * ts
        #print(miss)
        print("miss:", miss < 0)

        #print(xs * ts.reshape(-1, 1))
        En = np.dot((miss < 0), xs * ts.reshape(-1, 1))
        #print("En:", En)

        w += En * lr

        print("w e:", w)

xs = np.array([[0,0], [0,1], [1,0], [1,1]])
xs = np.hstack([xs, np.ones((xs.shape[0],1))])
ts = np.array([-1, -1, -1, 1])

np.random.seed(0)
w = np.random.normal(0., 1, (3))
print("w s:", w)

iteration = 100

lr = 0.1

learn(xs, ts, w, lr)

print(forward(xs, w))
