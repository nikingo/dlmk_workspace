import math
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
import keras.optimizers as opt
#To use confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import decomposition as pca

#from myPackage import preprpcess as pp


(x_train, y_train), (x_test, y_test) = mnist.load_data()    #read img data of 28*28

# plt.figure(0)
# for imgInd in range(100):
#     plt.subplot(10, 10, imgInd + 1)
#     plt.imshow(x_train[imgInd], cmap=cm.get_cmap("gray_r"), interpolation="nearest")
#     plt.axis("off")
#plt.show()

# 画像を1次元化
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 画素を0~1の範囲に変換(正規化)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 正解ラベルをone-hot-encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

pcaIns = pca.PCA(n_components=512)
x_train = pcaIns.fit_transform(x_train)
x_test = pcaIns.transform(x_test)

#print(pcaIns.explained_variance_ratio_)

model = Sequential()
model.add(Dense(units=128, activation="relu", input_dim=x_train.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu", input_dim=128))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation="relu", input_dim=64))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation="relu", input_dim=32))
model.add(Dropout(0.1))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer=opt.Adamax(), loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size=100, epochs=30, verbose=0, validation_split=0.1)

#score = model.evaluate(x_test, y_test)
predictedLabels = model.predict_classes(x_test, verbose=0)

#print(predictedLabels)
#print(y_test)

y_test_labels = np.argmax(y_test, 1)
#print(y_test_labels)

print(confusion_matrix(y_test_labels, predictedLabels))
print(classification_report(y_test_labels, predictedLabels))

plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
