import os
import glob
import cv2
import numpy as np
import datetime
import pickle

base_path = os.path.abspath(os.getcwd())
data_dir = "cifar-10-batches-py"
data_path = os.path.join(base_path, data_dir)

labels = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

xs = np.ndarray([0, 32, 32, 3], dtype=np.float32)
ys = np.ndarray([0,], dtype=np.int)

for ind in range(5):

    data_file = 'data_batch_' + str(ind + 1)

    with open(os.path.join(data_path, data_file), 'rb') as f:
        datas = pickle.load(f, encoding='bytes')

    x = datas[b'data']
    y = np.array(datas[b'labels'], dtype=np.int)

    xs = np.vstack((xs, x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)))
    ys = np.hstack((ys, y))

cv2.imshow(str(y[-1]), xs[0].astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


print(xs.shape)
print(ys.shape)