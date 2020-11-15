import tensorflow as tf
from PIL import Image
import numpy as np
import shutil
import os
from data_manager import DataManager
from data_manager2 import DataManager2
from model import Model
from config import *
import utils
from datetime import datetime
import csv



class Agent(object):
    def __init__(self, param):

        self.__sess = tf.Session()
        self.__Param = param
        self.init_datasets()  # 初始化数据管理器
        self.model = Model(self.__sess, self.__Param)  # 建立模型
        self.logger = utils.get_logger(param["Log_dir"])
        self.seg_count_total_pics = 0 # 图片总数
        self.seg_count_TP = 0  # 分割网络输出-真正例
        self.seg_count_FP = 0  # 分割网络输出-假正例
        self.seg_count_TN = 0  # 分割网络输出-真反例
        self.seg_count_FN = 0  # 分割网络输出-假反例
        self.csv_file = open(csv_name,'w',newline="")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow(["index", "filename", "TP",  "FP", "FN", "TN", "accuracy", "precision", "recall"])
    def run(self):
        if self.__Param["mode"] is "training":
            train_mode = self.__Param["train_mode"]
            self.train(train_mode)
        elif self.__Param["mode"] is "testing":
            self.test()
        elif self.__Param["mode"] is "savePb":
            self.pred()
        else:
            print("got a unexpected mode ,please set the mode  'training', 'testing' or 'savePb' ")

    def init_datasets(self):
        if self.__Param["mode"] is "training":
            self.Positive_data_list, self.Negative_data_list = self.listData1(self.__Param["data_dir"])
            self.DataManager_train_Positive = DataManager(self.Positive_data_list, self.__Param)
            self.DataManager_train_Negative = DataManager(self.Negative_data_list, self.__Param)
        elif self.__Param["mode"] is "testing":
            print("testing......initdatasets")
            self.Positive_data_list, self.Negative_data_list = self.listData1(self.__Param["data_dir"])
            self.DataManager_test_Positive = DataManager(self.Positive_data_list, self.__Param, shuffle=False)
            self.DataManager_test_Negative = DataManager(self.Negative_data_list, self.__Param, shuffle=False)
        elif self.__Param["mode"] is "savePb":
            self.data_list,data_size=self.listData(self.__Param["data_dir"])
            self.DataManager_data = DataManager2(self.data_list,self.__Param,shuffle=False)
        else:
            raise Exception('got a unexpected  mode ')

    def pred(self):
        visualization_dir = "./visualization/312"
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        with self.__sess.as_default():
            DataManager2 = self.DataManager_data # ??存疑
            for batch in range(DataManager2.number_batch):
                img_batch, file_name_batch = self.__sess.run(DataManager2.next_batch)  # 应该没有label了
                mask_batch, output_batch = self.__sess.run([self.model.mask, self.model.output_class],
                                                            feed_dict={self.model.Image: img_batch, })
                self.visualization2(img_batch, mask_batch, file_name_batch,
                                    save_dir=visualization_dir)

    def visualization2(self, img_batch, mask_batch, filenames, save_dir="./visualization"):
        # anew a floder to save visualization
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i, filename in enumerate(filenames):
            filename = str(filename).split("'")[-2].replace("/", "_")
            mask = np.array(mask_batch[i]).squeeze(2) * 255
           # image=np.array(img_batch[i]).squeeze(2)
           # img_visual=utils.concatImage([image,mask])
            im = Image.fromarray(mask)
            im = im.convert("L")
            visualization_path = os.path.join(save_dir, filename)
        #    img_visual.save(visualization_path)
            im.save(visualization_path)

    def train(self, mode):
        if mode not in ["segment", "decision", "total"]:
            raise Exception('got a unexpected  training mode ,options :{segment,decision}')
        with self.__sess.as_default():
            self.logger.info('start training {} net'.format(mode))
            for i in range(self.model.step, self.__Param["epochs_num"] + self.model.step):
                # epoch start
                iter_loss = 0
                for batch in range(self.DataManager_train_Positive.number_batch):
                    # batch start
                    for index in range(2):
                        # corss training the positive sample and negative sample
                        if index == 0:
                            img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(
                                self.DataManager_train_Positive.next_batch)
                        else:
                            img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(
                                self.DataManager_train_Negative.next_batch)
                        loss_value_batch = 0

                        if mode == "segment":
                            _, loss_value_batch = self.__sess.run([self.model.optimize_segment, self.model.loss_pixel],
                                                                  feed_dict={self.model.Image: img_batch,
                                                                             self.model.PixelLabel: label_pixel_batch})
                        iter_loss += loss_value_batch
                        # 可视化
                        if i % self.__Param["valid_frequency"] == 0 and i > 0:
                            mask_batch = self.__sess.run(self.model.mask, feed_dict={self.model.Image: img_batch})
                            save_dir = "./visualization/training_epoch-{}".format(i)
                            self.visualization(img_batch, label_pixel_batch, mask_batch, file_name_batch, save_dir)
                print("epoch: ", self.model.step, "train_mode: ", mode, "loss: ", iter_loss)
                self.logger.info('epoch:[{}] ,train_mode:{}, loss: {}'.format(self.model.step, mode, iter_loss))
                # 保存模型
                if i % self.__Param["save_frequency"] == 0 or i == self.__Param["epochs_num"] + self.model.step - 1:
                    self.model.save()
                # #验证
                # if i % self.__Param["valid_frequency"] == 0 and i>0:
                #  self.valid()
                self.model.step += 1
    def test(self):
        visualization_dir = "./visualization/test"
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        with self.__sess.as_default():
            self.logger.info('start testing')

            self.logger.info('Threshold: %.2f', SEGMENT_OUTPUT_THRESHOLD)
            self.logger.info('pixels of single pic: %d', IMAGE_SIZE[0]/8 * IMAGE_SIZE[1]/8)
            self.logger.info('===========Accuracy,Precision,Recall of single file============')

            print("start testing")
            count = 0
            count_TP = 0  # 真正例
            count_FP = 0  # 假正例
            count_TN = 0  # 真反例
            count_FN = 0  # 假反例
            DataManager = [self.DataManager_test_Positive, self.DataManager_test_Negative]
            for index in range(2):
                print("DataManager[index].number_batch:", DataManager[index].number_batch)
                for batch in range(DataManager[index].number_batch):
                    #返回值：图片、二值化图片、标签、图片名称
                    img_batch, label_pixel_batch, label_batch, file_name_batch, = self.__sess.run(
                        DataManager[index].next_batch)

                    #输入：图片
                    #输出：mask----分割网络后的结果
                    #输出：output_class----决策网络后的结果0/1
                    mask_batch, features, output_batch , logits_pixel = self.__sess.run([self.model.mask,
                        self.model.features, self.model.output_class, self.model.logits_pixel], feed_dict={self.model.Image: img_batch, })
                    #科学计数法
                    #np.set_printoptions(suppress=True)
                    #完全打印
                    np.set_printoptions(threshold=np.inf)

                    #print("mask_batch,,,,,,,,:", mask_batch)
                    print("img.shape,,,,,,,,:", img_batch.shape)
                    print("label_pixel_batch.shape,,,,,,,,:", label_pixel_batch.shape)
                    print("mask_batch.shape,,,,,,,,:", mask_batch.shape)
                    print("features.shape,,,,,,,,:", features.shape)
                    print("logits_pixel.shape,,,,,,,,:", logits_pixel.shape)
                    #print("logits,,,,,,,,:", logits_pixel)
                    #print("output_batch,,,,,,,,:", output_batch)
                    print("output_batch.shape,,,,,,,,:", output_batch.shape)
                    self.visualization(img_batch, label_pixel_batch, mask_batch, file_name_batch,
                                   save_dir=visualization_dir)

                    for i, filename in enumerate(file_name_batch):
                        print(filename)
                        count += 1
                        if label_batch[i] == 1 and output_batch[i] == 1:
                            count_TP += 1
                        elif label_batch[i] == 1:
                            count_FN += 1
                        elif output_batch[i] == 1:
                            count_FP += 1
                        else:
                            count_TN += 1

            self.caculate_total(IMAGE_SIZE[0]/8 * IMAGE_SIZE[1]/8)

            # 准确率
            accuracy = (count_TP + count_TN) / count
            # 查准率
            prescision = count_TP / (count_TP + count_FP)
            # 查全率
            recall = count_TP / (count_TP + count_FN)
            self.logger.info("output of decision network:===========================")
            self.logger.info("total number of samples = {}".format(count))
            self.logger.info("positive = {}".format(count_TP + count_FN))
            self.logger.info("negative = {}".format(count_FP + count_TN))
            self.logger.info("TP = {}".format(count_TP))
            self.logger.info("NP = {}".format(count_FP))
            self.logger.info("TN = {}".format(count_TN))
            self.logger.info("FN = {}".format(count_FN))
            self.logger.info("accuracy(准确率) = {:.4f}".format((count_TP + count_TN) / count))
            self.logger.info("prescision（查准率） = {:.4f}".format(prescision))
            self.logger.info("recall（查全率） = {:.4f}".format(recall))
            self.logger.info("the visualization saved in {}".format(visualization_dir))
    def valid(self):
        pass

    # 输入三张图片：原图、标签图、检测图(512, 1408) (64, 176) (64, 176)
    # 拼接后保存(1536, 1408)
    # 这里输入的是列表
    def visualization(self, img_batch, label_pixel_batch, mask_batch, filenames, save_dir="./visualization"):
        # anew a floder to save visualization
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(filenames)
        #print(mask_batch)
        for i, filename in enumerate(filenames):
            # 将文件夹与文件名称合并
            filename = str(filename).split("'")[-2].replace("/", "_")
            #每个像素值×255,还原为真真实像素-----分割网络预测值每个像素0～255
            mask_temp = np.array(mask_batch[i]).squeeze(2)
            #mask_temp[mask_temp > 0.1] = 255
            mask = mask_temp * 255
            #真实像素0～255
            image = np.array(img_batch[i]).squeeze(2)
            #标签像素0/255
            label_pixel = np.array(label_pixel_batch[i]).squeeze(2) * 255
            #(512, 1408) (64, 176) (64, 176)
            #后两张图双线性插值法扩大到与第一张一样大
            img_visual = utils.concatImage([image, label_pixel, mask])

            print("label_pixel type: ", label_pixel.shape, type(label_pixel))
            print("mask_batch[i]  type: ", mask_temp.shape, type(mask_temp))

            #计算单张图片、所有图片的准确率（Accuracy）精确率（Precision）召回率（Recall）
            self.caculate_single(filename, mask_temp, label_pixel)
            #720896 11264 11264 (1536, 1408)
            print(image.size, label_pixel.size, mask.size, img_visual.size)

            visualization_path = os.path.join(save_dir, filename)
            img_visual.save(visualization_path)



    def listData(self, data_dir):
        """# list the files  of  the currtent  floder of  'data_dir'  ,subfoders are not included.
        :param data_dir:
        :return:  list of files
        """
        data_list = os.listdir(data_dir)
        data_list = [x[2] for x in os.walk(data_dir)][0]
        data_size = len(data_list)
        return data_list, data_size

    def listData1(self, data_dir, test_ratio=0.4, positive_index=POSITIVE_KolektorSDD):
        """ this function is designed for the Dataset of KolektorSDD,
            the positive samples and negative samples will be divided into two lists
        :param  data_dir:  the  data folder   of KolektorSDD
        :param  test_ratio: the proportion of test set
        :param positive_index:   the  list  of  index of   every subfolders' positive samples
        :return:    the list of  the positive samples and the list of negative samples
        """


        #print("data_dir", data_dir)
        example_dirs = [x[1] for x in os.walk(data_dir)][0]
        #print("example_dirs", example_dirs)
        example_lists = {os.path.basename(x[0]): x[2] for x in os.walk(data_dir)}
        #print("example_lists", example_lists)
        train_test_offset = np.floor(len(example_lists) * (1 - test_ratio))
        #print("len of example_lists", len(example_lists))
        #print("train_test_offset", train_test_offset)

        Positive_examples_train = []
        Negative_examples_train = []
        Positive_examples_valid = []
        Negative_examples_valid = []
        for i in range(len(example_dirs)):
            example_dir = example_dirs[i]
            example_list = example_lists[example_dir]
            # 过滤label图片
            example_list = [item for item in example_list if "label" not in item]
            #print("no labels: ")
            #print(example_list)
            # 训练数据
            #print("i, train_test_offset")
            #print(i, train_test_offset)
            if i < train_test_offset:
                for j in range(len(example_list)):
                    example_image = example_dir + '/' + example_list[j]
                    example_label = example_image.split(".")[0] + "_label.bmp"
                    # 判断是否是正样本
                    #index = example_list[j].split(".")[0][-1]
                    index = example_list[j].split(".")[0]
                    #print("index: ")
                    #print(index)
                    if index in positive_index[i]:
                        Positive_examples_train.append([example_image, example_label])
                    else:
                        Negative_examples_train.append([example_image, example_label])
            else:
                for j in range(len(example_list)):
                    example_image = example_dir + '/' + example_list[j]
                    example_label = example_image.split(".")[0] + "_label.bmp"
                    #index = example_list[j].split(".")[0][-1]
                    index = example_list[j].split(".")[0]
                    if index in positive_index[i]:
                        Positive_examples_valid.append([example_image, example_label])
                    else:
                        Negative_examples_valid.append([example_image, example_label])

            # print("Positive_examples_train:")
            # print(Positive_examples_train)
            # print("Negative_examples_train:")
            # print(Negative_examples_train)
            # print("Positive_examples_valid:")
            # print(Positive_examples_valid)
            # print("Negative_examples_valid:")
            # print(Negative_examples_valid)

        if self.__Param["mode"] is "training":
            return Positive_examples_train, Negative_examples_train
        if self.__Param["mode"] is "testing":
            return Positive_examples_valid, Negative_examples_valid

    # 计算单张图片的准确率（Accuracy）精确率（Precision）召回率（Recall）
    def caculate_single(self, filename, detection, label):
        count_TP = 0  # 真正例
        count_FP = 0  # 假正例
        count_TN = 0  # 真反例
        count_FN = 0  # 假反例

        #图片总数加1
        self.seg_count_total_pics += 1
        print("total pics: ", self.seg_count_total_pics)

        #二值化，  概率>0?255:0
        detection = np.where(detection > SEGMENT_OUTPUT_THRESHOLD, 255, 0)
        #去除单维，转列表
        detection = detection.reshape(-1, detection.size).squeeze().tolist()
        #去除单维，转列表
        count = label.size
        label = label.reshape(-1, count).squeeze().tolist()
        #print("detection size: ", type(detection), len(detection), detection)
        #print("label size: ", type(label),len(label), label)

        for i in range(len(detection)):
            if label[i] == 255 and detection[i] == 255:
                count_TP += 1
                self.seg_count_TP += 1
            elif label[i] == 255:
                count_FN += 1
                self.seg_count_FN += 1
            elif detection[i] == 255:
                count_FP += 1
                self.seg_count_FP += 1
            else:
                count_TN += 1
                self.seg_count_TN += 1

        print("count_TP, count_FN, count_FP, count_TN:", count_TP, count_FN, count_FP, count_TN)

        # 准确率(Accuracy)
        accuracy = (count_TP + count_TN) / count
        # 查准率(Precision)
        if (count_TP + count_FP) == 0:
            prescision = 0
        else:
            prescision = count_TP / (count_TP + count_FP)

        # 查全率(Recall)
        if (count_TP + count_FN) == 0:
            recall = 0
        else:
            recall = count_TP / (count_TP + count_FN)

        print("accuracu, precision, recall of single file: =====================")
        print("filename:", filename)
        print("total number of samples = {}".format(count))
        print("positive = {}".format(count_TP + count_FN))
        print("negative = {}".format(count_FP + count_TN))
        print("TP = {}".format(count_TP))
        print("FP = {}".format(count_FP))
        print("TN = {}".format(count_TN))
        print("FN = {}".format(count_FN))
        print("accuracy(准确率) = {:.4f}".format(accuracy))
        print("prescision（查准率） = {:.4f}".format(prescision))
        print("recall（查全率） = {:.4f}".format(recall))

        self.csv_writer.writerow([self.seg_count_total_pics, filename, count_TP, count_FP,  count_FN,
                            count_TN,round(accuracy,4), round(prescision,4), round(recall, 4)])
        self.logger.info('index: %d %s Accuracy: %.4f Precision: %.4f Recall: %.4f', self.seg_count_total_pics,
                             filename, accuracy, prescision, recall)

    # 计算所有图片的准确率（Accuracy）精确率（Precision）召回率（Recall）
    def caculate_total(self, pic_pixels):

        if self.seg_count_total_pics == 0:
            print("no pics! ")
            return

        count = self.seg_count_total_pics * pic_pixels

        # 准确率(Accuracy)
        accuracy = (self.seg_count_TP + self.seg_count_TN) / count
        # 查准率(Precision)
        if (self.seg_count_TP + self.seg_count_FN) == 0:
            prescision = 0
        else:
            prescision = self.seg_count_TP / (self.seg_count_TP + self.seg_count_FP)
        # 查全率(Recall)
        if (self.seg_count_TP + self.seg_count_FN) == 0:
            recall = 0
        else:
            recall = self.seg_count_TP / (self.seg_count_TP + self.seg_count_FN)


        print("accuracu, precision, recall of total files: ======================")
        print("total number of samples = {}".format(self.seg_count_total_pics))
        print("total number of pixels = {}".format(count))
        print("positive = {}".format(self.seg_count_TP + self.seg_count_FN))
        print("negative = {}".format(self.seg_count_FP + self.seg_count_TN))
        print("TP = {}".format(self.seg_count_TP))
        print("FP = {}".format(self.seg_count_FP))
        print("TN = {}".format(self.seg_count_TN))
        print("FN = {}".format(self.seg_count_FN))
        print("accuracy(准确率) = {:.4f}".format(accuracy))
        print("prescision（查准率） = {:.4f}".format(prescision))
        print("recall（查全率） = {:.4f}".format(recall))

        self.logger.info("output of decision network:===========================")
        self.logger.info('===========Accuracy,Precision,Recall of total files============')
        self.logger.info('Number of pics: %d', self.seg_count_total_pics)
        self.logger.info('pixels of total pics: %d', count)
        self.logger.info('TP: %d FP: %d TN: %d FN: %d', self.seg_count_TP,
                             self.seg_count_FP, self.seg_count_TN, self.seg_count_FN)
        self.logger.info('Accuracy : %.4f', accuracy)
        self.logger.info('Precision: %.4f', prescision)
        self.logger.info('Recall: %.4f', recall)

        self.csv_writer.writerow([self.seg_count_total_pics, "total", self.seg_count_TP,  self.seg_count_FP,
                    self.seg_count_FN,  self.seg_count_TN, round(accuracy, 4), round(prescision, 4), round(recall,4)])
        self.csv_writer.close()





