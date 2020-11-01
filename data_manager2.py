import re
import os
import numpy as np
import  cv2
from config import *
from scipy.misc import imread, imresize, imsave
from random import shuffle
import  tensorflow as tf

class DataManager2(object):


    def __init__(self, dataList, param, shuffle=True):
        """
        """
        self.shuffle = shuffle
        self.data_list = dataList
        self.data_size = len(dataList)
        self.data_dir = param["data_dir"]
        self.epochs_num = param["epochs_num"]
        self.batch_size = param["batch_size"]
        self.number_batch = int(np.floor(len(self.data_list) / self.batch_size))
        self.next_batch = self.get_next()


    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size * 3 + 200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch


    def generator(self):
        for index in range(len(self.data_list)):
            file_basename_image = self.data_list[index]
            image_path = os.path.join(self.data_dir, file_basename_image)
            image = self.read_data(image_path)
            image = (np.array(image[:, :, np.newaxis]))
            yield image, file_basename_image


    def read_data(self, data_name):
        img = cv2.imread(data_name, 0)  # /255.#read the gray image
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        # img = img.swapaxes(0, 1)
        # image = (np.array(img[:, :, np.newaxis]))
        return img

