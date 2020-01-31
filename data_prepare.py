import os, glob
import pickle

import numpy as np

import random

import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
import torchvision.transforms as transforms

import torch

class data_handler():
    def __init__(self, pretrained, input_path, output_path, batch_size):
        super(data_handler, self).__init__()

        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size

        train_samples, test_samples, val_samples = self.data_loader(pretrained)

        self.train_samples = self.batchify(train_samples)
        self.test_samples = self.batchify(test_samples)
        self.val_samples = self.batchify(val_samples)

    def data_loader(self, pretrained):
        if pretrained:
            data_file = os.path.join(self.output_path, '%s.pkl')
            with tf.gfile.GFile(data_file % 'train', 'rb') as f:
                train_samples = pickle.load(f)

            with tf.gfile.GFile(data_file % 'test', 'rb') as f:
                test_samples = pickle.load(f)

            with tf.gfile.GFile(data_file % 'val', 'rb') as f:
                val_samples = pickle.load(f)
        else:
            flower_types = [d for d in os.listdir(self.input_path) if os.path.isdir(self.input_path + "/"+ d)]

            train_samples = {'images':[], 'labels':[]}
            val_samples = {'images':[], 'labels':[]}
            test_samples = {'images':[], 'labels':[]}

            class_dic = {}

            for cl in flower_types:
                if cl not in class_dic:
                    class_dic[cl] = len(class_dic)

                files = [f for f in glob.glob(self.input_path + cl + "/*.jpg")]
                test_split = len(files) * 70 // 100
                val_split = len(files) * 90 // 100

                for i, file_name in enumerate(files):
                    current_img = imread(file_name, flatten=False)
                    if current_img.shape != ():
                        current_img = imresize(current_img, [64, 64], interp='bilinear', mode=None)
                        current_img = np.asarray(current_img).astype('float32')
                        max_pix_val = current_img.max()
                        current_img /= max_pix_val

                    if i < test_split:
                        train_samples['images'].append(current_img)
                        train_samples['labels'].append(class_dic[cl])
                    elif i > test_split and i < val_split:
                        test_samples['images'].append(current_img)
                        test_samples['labels'].append(class_dic[cl])
                    else:
                        val_samples['images'].append(current_img)
                        val_samples['labels'].append(class_dic[cl])

            write_path = self.output_path + "/train.pkl"
            with tf.gfile.GFile(write_path, 'w') as f:
                pickle.dump(train_samples, f)

            write_path = self.output_path + "/val.pkl"
            with tf.gfile.GFile(write_path, 'w') as f:
                pickle.dump(val_samples, f)

            write_path = self.output_path + "/test.pkl"
            with tf.gfile.GFile(write_path, 'w') as f:
                pickle.dump(test_samples, f)

        return train_samples, test_samples, val_samples

    def batchify(self, data):
        all_batches = []

        # shuffle the data
        l = list(zip(data['images'], data['labels']))
        random.shuffle(l)
        data['images'], data['labels'] = zip(*l)

        # batchify
        i = 0
        while i + self.batch_size < len(data['images']):
            batch_temp = torch.zeros(self.batch_size, 3, 64, 64)

            images = data['images'][i:i+self.batch_size]
            labels = torch.LongTensor(data['labels'][i:i+self.batch_size])

            images = [torch.from_numpy(img).float() for img in images]

            for j, img in enumerate(images):
                img = img.transpose(0,2).view(1, 3, 64, 64)
                batch_temp[j] = img

            all_batches.append([batch_temp, labels])

            i += self.batch_size

        return all_batches


