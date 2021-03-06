import math
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb
import argparse
import cv2
from glob import glob
import os

from prepare import data_load
from prepare import get_shuffled_batch_ind

import torch
import torchvision
import torch.nn.functional as F

num_classes = 2
img_height, img_width = 64, 64
GPU = False
torch.manual_seed(0)

device = torch.device("cuda" if GPU else "cpu")

class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = torch.nn.BatchNorm2d(32)
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = torch.nn.BatchNorm2d(64)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = torch.nn.BatchNorm2d(64)
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = torch.nn.BatchNorm2d(128)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = torch.nn.BatchNorm2d(128)
        self.conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = torch.nn.BatchNorm2d(256)
        self.conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = torch.nn.BatchNorm2d(256)
        self.fc1 = torch.nn.Linear(img_height//16 * img_width//16 * 256, 512)
        #self.fc1_d = torch.nn.Dropout2d()
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc_out = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, img_height//16 * img_width//16 * 256)
        x = F.relu(self.fc1(x))
        #x = self.fc1_d(x)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class Simplenet(torch.nn.Module):
    def __init__(self):
        super(Simplenet, self).__init__()

        self.conv1_1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1_1 = torch.nn.BatchNorm2d(16)
        self.conv1_2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(16)
        self.conv2_1 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_1 = torch.nn.BatchNorm2d(32)
        self.conv2_2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2_2 = torch.nn.BatchNorm2d(32)
        self.fc1 = torch.nn.Linear(img_height//4 * img_width//4 * 32, 64)
        #self.fc1_d = torch.nn.Dropout2d()
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc_out = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, img_height//4 * img_width //4 * 32)
        x = F.relu(self.fc1(x))
        #x = self.fc1_d(x)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

def train_net(model, opt, data_path):

    xs, ys = data_load(data_path, 64, 64, hflip=True, vflip=True, rot=[angle for angle in range(0,360,10)])

    ind_batch = get_shuffled_batch_ind(len(ys), 32, 10)
    iter_per_epoch = len(ys) // 32

    running_loss = 0
    for i, (batch_xs, batch_ys) in enumerate(zip(xs[ind_batch], ys[ind_batch])):

        inputs = torch.tensor(batch_xs, dtype=torch.float).to(device)
        labels = torch.tensor(batch_ys, dtype=torch.long).to(device)
        
        # 勾配情報をリセット
        opt.zero_grad()
        
        # 順伝播
        outputs = model(inputs)
        
        # コスト関数を使ってロスを計算する
        loss = F.cross_entropy(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        opt.step()
        
        running_loss += loss.item()
            
        if i % iter_per_epoch == iter_per_epoch - 1:
            print('%d loss: %.3f' % (i + 1, running_loss / 100))
            running_loss = 0.0

def test_net(model, data_path):

    correct = 0.0
    total = 0

    xs, ys = data_load(data_path, 64, 64, hflip=False, vflip=False)

    model.eval()
    for i in range(len(ys)):

        input_shape = xs[i].shape
        #print(input_shape)
        x = xs[i].reshape(1, input_shape[0], input_shape[1],input_shape[2])

        inputs = torch.tensor(x, dtype=torch.float).to(device)
        labels = torch.tensor(ys[i], dtype=torch.long).to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        print('label:', labels.numpy(), 'out:', predicted.numpy())
        
        #total += labels.size(0)
        #print('Accuracy %d / %d = %f' % (correct, total, correct / total))


#model = Mynet()
model = Simplenet()
model.to(device)

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

base_path = os.path.abspath(os.path.dirname(__file__))
data_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\train\\images"
test_dir = "..\\..\\DeepLearningMugenKnock\\Dataset\\test\\images"
data_path = os.path.join(base_path, data_dir)
test_path = os.path.join(base_path, test_dir)

train_net(model, opt, data_path)
#model.load_state_dict(torch.load(os.path.join(base_path, "cnn.pt")))
test_net(model, test_path)
#torch.save(model.state_dict(), os.path.join(base_path, "cnn.pt"))