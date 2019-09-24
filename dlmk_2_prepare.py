import os
import glob
import cv2
import numpy as np
import datetime

def data_load(dir_path, img_height, img_width, hflip=False, vflip=False, rot=[]):

    file_list = glob.glob(dir_path + '\\**\\*')

    xs = []
    ys = []

    class_label = ['akahara', 'madara']

    for file_path in file_list:
        img = cv2.imread(file_path)
        img_resize = cv2.resize(img, (img_height, img_width)).astype(np.float32)
        img_resize /= 255. 
        xs.append(img_resize)
        if class_label[0] in file_path:
            y = 0
        else:
            y = 1

        ys.append(y)

        if hflip:
            xs.append(cv2.flip(img_resize, 1))
            ys.append(y)
        
        if vflip:
            xs.append(cv2.flip(img_resize, 0))
            ys.append(y)

        _h, _w, _c = img.shape
        max_side = max(_h, _w)
        min_side = min(_h, _w)
        tmp = np.zeros((max_side, max_side, _c))
        tx = int((max_side - _w) / 2)
        ty = int((max_side - _h) / 2)
        tmp[ty: ty+_h, tx: tx+_w] = img.copy()
        resize_offset = ((int(max_side / 2) - int(min_side / 2)), int(max_side / 2) - int(min_side / 2))
        for angle in rot:
            matrix = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, 1.0)
            image_rot = cv2.warpAffine(tmp, matrix, (max_side, max_side))
            img_resize = image_rot[resize_offset[0]:resize_offset[0]+min_side, resize_offset[1]:resize_offset[1]+min_side]
            img_resize = cv2.resize(img_resize, (img_height, img_width)).astype(np.float32)
            img_resize /= 255. 
            print(img_resize.shape)
            xs.append(img_resize)
            ys.append(y)

        # center = (int(img_height / 2), int(img_width / 2))
        # for angle in rot:
        #     matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        #     image_rot = cv2.warpAffine(img_resize, matrix, (img_height, img_width))
        #     xs.append(image_rot)
        #     ys.append(y)
        

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ys, file_list

def get_shuffled_batch_ind(data_size, batch_size, epoch):

    mbi = 0
    train_ind = np.arange(data_size)

    data_rest_num = 0
    data_carry_num = 0
    ind_batch = []

    np.random.shuffle(train_ind)

    for _ in range(epoch):
        train_ind_part = train_ind[data_carry_num:]
        iteration = len(train_ind_part) // batch_size
        batch_num_per_epoch = 0
        
        for batch_ind in range(iteration):
            batch_start_ind = batch_ind * batch_size
            ind_batch.append(train_ind_part[batch_start_ind:batch_start_ind + batch_size].copy())
            batch_num_per_epoch += 1
        
        data_rest_num = len(train_ind_part) - batch_num_per_epoch * batch_size

        if data_rest_num > 0:
            rest_data = train_ind_part[-data_rest_num:].copy()
            data_carry_num = (batch_size - data_rest_num) % batch_size
            np.random.shuffle(train_ind)
            ind_batch.append(np.hstack((rest_data, train_ind[:data_carry_num])))
        else:
            np.random.shuffle(train_ind)

    return ind_batch



base_path = os.path.abspath(os.getcwd())
data_dir = "..\\DeepLearningMugenKnock\\Dataset\\train\\images"
data_path = os.path.join(base_path, data_dir)

img_height, img_width = 64, 64

xs, ys, file_list = data_load(data_path, img_height, img_width, hflip=True, vflip=True, rot=[45, -45])

ind_batch = get_shuffled_batch_ind(len(ys), 3, 3)

print(ind_batch)

if True:

    date_str = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    test_out_dir = "test_img_dir\\q2_" + date_str
    test_output_path = os.path.join(base_path, test_out_dir)
    if not os.path.isdir(test_output_path):
        os.mkdir(test_output_path)

    for i, (img, label) in enumerate(zip(xs, ys)):
        cv2.imwrite(os.path.join(test_output_path, str(i) + "_" + str(label) + ".jpg"), (img * 255).astype(np.uint8).transpose(1,2,0))