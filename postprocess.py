import cv2
from PIL import Image
from os import listdir
import tensorflow as tf
import os
import numpy as np
import time

Image.MAX_IMAGE_PIXELS = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SPLIT_WIDTH = 512
SPLIT_HEIGHT = 1408
SPLIT_ROWS = 10
SPLIT_COLS = 28

THRESHOLD_BINARYZATION = 0.1
THRESHOLD_POSTPROCESS = 0.4
THRESHOLD_COLSPIX = 0.25

SPLIT_WIDTH_POSTPROCESS = 4
SPLIT_HEIGHT_POSTPROCESS = 22
SPLIT_ROWS_POST_PROCESS = 8
SPLIT_COLS_POST_PROCESS = 16

#function: 读取大图，先转为灰度并保存，然后分割成小图
#          分割的行数、列数向下取整，会丢失一部分边界像素
#bmpName: 待分割图片名称(e.g. 222.bmp)
#bmpgrayName: 保存灰度图的名称(e.g. 222_gray.bmp)
#bmpPath: 保存分割完的图片的路径(e.g. ./222/)
def splitBMP(bmpName, bmpgrayName, bmpsPath):
    img = cv2.imread(bmpName, 0)
    cv2.imwrite(bmpgrayName, img)
    img = Image.open(bmpgrayName)
    img_size = img.size
    m = img_size[0]   
    n = img_size[1]     
    SPLIT_COLS = int(m / SPLIT_WIDTH)
    SPLIT_ROWS = int(n / SPLIT_HEIGHT)     
    
    pic_index = 0
    for x in range(SPLIT_COLS):
        for y in range(SPLIT_ROWS): 
            rect = (x*SPLIT_WIDTH, y*SPLIT_HEIGHT, (x+1)*SPLIT_WIDTH,(y+1)*SPLIT_HEIGHT)
            print(rect)
            region = img.crop(rect)
            pic_index = pic_index + 1
            region.save(bmpsPath + str(pic_index).zfill(4) + ".bmp")
            print("save pic--> ", pic_index)

    #print("shape: ", img.shape[:2][::1])
    print("rows, cols: ", SPLIT_ROWS,SPLIT_COLS)
    print("total pic count: ", pic_index)
    print("split done.")



#fuction： 组合大图
#bmpsPath: 待拼接小图的路径(e.g. ./222/)
#bmpgrayJointName: 拼接后的图片名称(e.g. 222_gray_all.bmp)
#rows:  大图中每列几个小图
#cols:  大图中每行几个小图
def jointBMP(bmpsPath, bmpgrayJointName, rows, cols):
    img_list = [Image.open(bmpsPath + fn) for fn in listdir(bmpsPath)]
    width, height = img_list[0].size
    #print(width, height, img_list[0].mode)
    result = Image.new(img_list[0].mode, (width*SPLIT_COLS, height*SPLIT_ROWS))
    print("dest bmp size: ", result.size)


    count = 0
    for i in range(cols):
        for j in range(rows):
            result.paste(img_list[count], (i*SPLIT_WIDTH, j*SPLIT_HEIGHT))
            count = count + 1
    
    result.save(bmpgrayJointName)
    print("joint done.")

#function: 识别单个图片
#pbPath: pb模型文件路径
#bmpPath: 待识别图片路径
#return: 64 * 176 大小的mask
def detectbmp(pbPath, bmpPath):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pbPath, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("Image:0")
            output_tensor_name = sess.graph.get_tensor_by_name("segment/Sigmoid:0")
            im = cv2.imread(bmpPath,0)
            image = (np.array(im[np.newaxis, :, :, np.newaxis]))
            out = sess.run(output_tensor_name, feed_dict={input_image_tensor: image})

            #print("out:{}".format(out))
            return np.squeeze(out)
            #cv2.imwrite("outbmp.bmp", out1*255)


#function: 识别整个文件夹图片, 放大到512×1408后再拼接起来
#pbPath: pb模型路径
#bmpsPath: 图片目录
#bmpDetectedPath： 识别后的图片目录
#bmpjointPath: 识别后拼接保存的图片目录
#rows:  大图中每列几个小图
#cols:  大图中每行几个小图
def detectbmps(pbPath, bmpsPath, bmpDetectedPath, bmpDetectedPath_nopost, bmpjointPath,
               bmpjointPath_nppost, cols, rows):
    start = time.clock()
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pbPath, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input_image_tensor = sess.graph.get_tensor_by_name("Image:0")
            output_tensor_name = sess.graph.get_tensor_by_name("segment/Sigmoid:0")

            img_list = [Image.open(bmpsPath + fn) for fn in listdir(bmpsPath)]
            width, height = img_list[0].size
            result = Image.new(img_list[0].mode, (width * cols, height * rows))
            result_nopost = Image.new(img_list[0].mode, (width * cols, height * rows))

            pic_index = 0
            for i in range(cols):
                for j in range(rows):
                    image = np.array(img_list[pic_index])
                    image = (np.array(image[np.newaxis, :, :, np.newaxis]))
                    interval_1 = time.clock()
                    out = sess.run(output_tensor_name, feed_dict={input_image_tensor: image})
                    interval_2 = time.clock()
                    print("detect time: ", interval_2 - interval_1)
                    print("detect-->", pic_index)
                    mask = np.squeeze(out)
                    mask_resize_nopost = cv2.resize(mask, (width, height))

                    mask = postprocess(mask, THRESHOLD_BINARYZATION, THRESHOLD_POSTPROCESS)
                    mask_resize = cv2.resize(mask, (width, height))

                    save_path = bmpDetectedPath + str(pic_index).zfill(4) + ".bmp"
                    save_path_nopost = bmpDetectedPath_nopost + str(pic_index).zfill(4) + ".bmp"

                    cv2.imwrite(save_path, mask_resize)
                    cv2.imwrite(save_path_nopost, mask_resize_nopost*255)

                    print("save-->", pic_index)
                    result.paste(Image.fromarray(np.uint8(mask_resize)), (i * width, j * height))
                    result_nopost.paste(Image.fromarray(np.uint8(mask_resize_nopost*255)), (i * width, j * height))
                    print("paste-->", pic_index)
                    pic_index = pic_index + 1

            end = time.clock()
            print("total time: ", end-start)
            result.save(bmpjointPath)
            result_nopost.save(bmpjointPath_nppost)
            print("joint done.")


#function: 后处理, 对图片分小块,对小块的里的像素先进行二值化(0-1),若1的个数>阈值,则该区域都为1,否则为0
#img: 待处理图片数据 np.arrary
#thread_binary： 二值化阈值
#thread_postprocess： 后处理阈值
def postprocess(img, thread_binary=0.1, thread_postprocess=0.5):
    image = ImageBinarization(img, thread_binary)
    image = Image.fromarray(np.uint8(image), mode='L')
    #4*16 22*8
    result = Image.new('L', (SPLIT_WIDTH_POSTPROCESS * SPLIT_COLS_POST_PROCESS,
                             SPLIT_HEIGHT_POSTPROCESS * SPLIT_ROWS_POST_PROCESS))

    crops_imgs = []
    index = 0
    for x in range(SPLIT_COLS_POST_PROCESS):
        for y in range(SPLIT_ROWS_POST_PROCESS):
            rect = (x*SPLIT_WIDTH_POSTPROCESS, y*SPLIT_HEIGHT_POSTPROCESS,
                    (x+1)*SPLIT_WIDTH_POSTPROCESS,(y+1)*SPLIT_HEIGHT_POSTPROCESS)
            region = image.crop(rect)
            index = index + 1
            crops_imgs.append(region)
            #print("index: ", index)

    print(len(crops_imgs))
    thr = SPLIT_WIDTH_POSTPROCESS * SPLIT_HEIGHT_POSTPROCESS * thread_postprocess
    print(thr)
    cols_count = 0
    cols_count_pix = 0
    for i in range(len(crops_imgs)):
        cols_count = cols_count + 1
        im = np.array(crops_imgs[i])
        #print(im)
        count_pix1 = sum(sum(im == 255))

        if count_pix1 >= thr:
            #im = np.where(im == 0, 255, 255)
            cols_count_pix = cols_count_pix + 1
            print("> thr: ", count_pix1)
        #else:
        #    #im = np.where(im == 255, 0, 0)

        if cols_count % SPLIT_ROWS_POST_PROCESS == 0:
            print("cols_count: ", cols_count, "thr: ", SPLIT_ROWS_POST_PROCESS*THRESHOLD_COLSPIX)
            print("cols_count_pixs: ", cols_count_pix)
            if cols_count_pix > SPLIT_ROWS_POST_PROCESS*THRESHOLD_COLSPIX:
                for i in range(cols_count - SPLIT_ROWS_POST_PROCESS, cols_count):
                    img = np.array(crops_imgs[i])
                    im = np.where(img == 0, 255, 255)
                    crops_imgs[i] = Image.fromarray(np.uint8(im), mode='L')
                    print("set 1")
            else:
                for i in range(cols_count - SPLIT_ROWS_POST_PROCESS, cols_count):
                    img = np.array(crops_imgs[i])
                    im = np.where(img == 255, 0, 0)
                    crops_imgs[i] = Image.fromarray(np.uint8(im), mode='L')
                    print("set 0")

            cols_count_pix = 0


    count = 0
    for i in range(SPLIT_COLS_POST_PROCESS):
        for j in range(SPLIT_ROWS_POST_PROCESS):
            result.paste(crops_imgs[count], (i * SPLIT_WIDTH_POSTPROCESS, j * SPLIT_HEIGHT_POSTPROCESS))
            #print("joint-->", count)
            count = count + 1

    return np.array(result)





def ImageBinarization(img, threshold=0.1):
    image = np.where(img > threshold, 255, 0)
    return image

if __name__ == '__main__':
    splitBMP("555.bmp", "./test/555/555_gray.bmp", "./test/555/")
    jointBMP("./test/555/", "./test/555_gray_joint.bmp", SPLIT_ROWS, SPLIT_COLS)
    detectbmps("ckp-174.pb", "./test/555", "./test/555_detected/", "./test/555_detected_nopost/",
               "./test/555_detected_jointed.bmp", "./test/555_detected_jointed_nopost.bmp", 28, 10)
    #img = detectbmp("ckp-174.pb","./222/0013.bmp")
    #cv2.imwrite("outbmp.bmp", img * 255)
    #detectbmps("ckp-174.pb", "./222/", "./222_detected/", "222_detected_jointed.bmp", 28, 10)
    #img = detectbmp("ckp-174.pb", "./222/0017.bmp")
    #cv2.imwrite("asd.bmp", img*255)
    #postprocess(img)



