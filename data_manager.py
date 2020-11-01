import re
import os
import numpy as np
import  cv2
from config import *
from scipy.misc import imread, imresize, imsave
from random import shuffle
import  tensorflow as tf

class DataManager(object):
    def __init__(self, dataList,param,shuffle=True):
        """
        """
        self.shuffle=shuffle
        self.data_list=dataList
        self.data_size=len(dataList)
        self.data_dir=param["data_dir"]
        self.epochs_num=param["epochs_num"]
        self.batch_size = param["batch_size"]
        self.number_batch = int(np.floor(len(self.data_list) /self.batch_size))
        self.next_batch=self.get_next()

    #从迭代器里获取数据
    #获取所有数据->复制×迭代次数->打乱数据->分批次一个一批->返回一批数据
    def get_next(self):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.int32,tf.int32, tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset = dataset.shuffle(self.batch_size*3+200)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        out_batch = iterator.get_next()
        return out_batch

    #输入图片列表
    #返回图片（改图片大小为指定大小）
    #返回图片（分辨率为1/8,二值化）及其标签（标签：有白点则为1）
    #返回图片名称
    #以上4项打包成迭代器返回（tf.float32, tf.int32,tf.int32, tf.string）
    def generator(self):
        for index in range(len(self.data_list)):
            file_basename_image,file_basename_label = self.data_list[index]
            image_path = os.path.join(self.data_dir, file_basename_image)
            label_path= os.path.join(self.data_dir, file_basename_label)
            image= self.read_data(image_path)
            label = self.read_data(label_path)
            label_pixel,label=self.label_preprocess(label)
            #变换，每个像素一个数组，先按行
            image = (np.array(image[:, :, np.newaxis]))
            label_pixel = (np.array(label_pixel[:, :, np.newaxis]))
            yield image, label_pixel,label, file_basename_image

    #输入图片
    #返回图片（改变为指定大小）灰度像素0-255
    def read_data(self, data_name):
        img = cv2.imread(data_name, 0)  # /255.#read the gray image
        img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
        # img = img.swapaxes(0, 1)
        # image = (np.array(img[:, :, np.newaxis]))
        return img

    #输入图片
    #返回图片（分辨率变为1/8,二值化,如果有白点则返回标签值1）
    def label_preprocess(self,label):
        label = cv2.resize(label, (int(IMAGE_SIZE[1]/8), int(IMAGE_SIZE[0]/8)))
        label_pixel=self.ImageBinarization(label)
        label=label.sum()
        if label>0:
            label=1
        return  label_pixel,label

    #输入图片
    #返回图片（二值化,像素值>1？1：0）
    def ImageBinarization(self,img, threshold=1):
        img = np.array(img)
        image = np.where(img > threshold, 1, 0)
        return image

    def label2int(self,label):  # label shape (num,len)
        # seq_len=[]
        target_input = np.ones((MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
        target_out = np.ones(( MAX_LEN_WORD), dtype=np.float32) + 2  # 初始化为全为PAD
        target_input[0] = 0  # 第一个为GO
        for j in range(len(label)):
            target_input[j + 1] = VOCAB[label[j]]
            target_out[j] = VOCAB[label[j]]
            target_out[len(label)] = 1
        return target_input, target_out

    def int2label(self,decode_label):
        label = []
        for i in range(decode_label.shape[0]):
            temp = ''
            for j in range(decode_label.shape[1]):
                if VOC_IND[decode_label[i][j]] == '<EOS>':
                    break
                elif decode_label[i][j] == 3:
                    continue
                else:
                    temp += VOC_IND[decode_label[i][j]]
            label.append(temp)
        return label


    def get_label(self,f):
        return  f.split('.')[-2].split('_')[1]
