import cv2
import copy
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import tensorflow as tf


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


#标签路径
lab_path = r"E:\paper2\CloudUnetv2\database\all\png"


def calculate_map(input_image, saved_model):


    test_image = np.expand_dims(input_image, axis=0)
    show_test_image = np.squeeze(test_image)

    decoded_img = saved_model.predict(test_image)[0]
    show_decoded_image = np.squeeze(decoded_img)


    return show_decoded_image

def calculate_score_threshold(input_map, groundtruth_image):

    binary_map = input_map
    # binary_map[binary_map < threshold] = 0
    # binary_map[binary_map == threshold] = 0
    # binary_map[binary_map > threshold] = 255
    # print(binary_map)
    # print("binary_map数组元素数据类型：", binary_map.dtype)  # 打印数组元素数据类型
    # print("binary_map数组元素总数：", binary_map.size)  # 打印数组尺寸，即数组元素总数
    # print("binary_map数组形状：", binary_map.shape)  # 打印数组形状
    # print("binary_map数组的维度数目", binary_map.ndim)  # 打印数组的维度数目

    groundtruth_image = groundtruth_image
    # print("groundtruth_image数组元素数据类型：", groundtruth_image.dtype)  # 打印数组元素数据类型
    # print("groundtruth_image数组元素总数：", groundtruth_image.size)  # 打印数组尺寸，即数组元素总数
    # print("groundtruth_image数组形状：", groundtruth_image.shape)  # 打印数组形状
    # print("groundtruth_image数组的维度数目", groundtruth_image.ndim)  # 打印数组的维度数目
    WIDTH = 320
    HIGTH = 320

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(WIDTH):
        for j in range(HIGTH):
            if (groundtruth_image[i, j] == 255 and binary_map[i, j][1] == 255):  # TP condition
                TP = TP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j][1] == 255)):  # FP condition
                FP = FP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j][1] == 0)):  # TN condition
                TN = TN + 1
            elif ((groundtruth_image[i, j] == 255) and (binary_map[i, j][1] == 0)):  # FN condition
                FN = FN + 1
    # print('TP', TP)
    # print('FP', FP)
    # print('TN', TN)
    # print('FN', FN)
    if float(TP+FN)==0 or float(FP + TN)==0:
        tpr = 0
        fpr = 0
    else:
        tpr = float(TP) / float(TP + FN)
        fpr = float(FP) / float(FP + TN)

    return (tpr, fpr)



TPR_values = {}
FPR_values = {}
b = 0
d = 0
n = 0

dbtype_list = os.listdir(r'E:\paper2\CloudUnetv2\img')
for dbtype in dbtype_list:
# for threshold in range(0, 101):

    # a = str(roc_values[threshold])
    a = str(dbtype)

    # 图像
    pred_path = r"E:\paper2\CloudUnetv2\img" + '/' + a +'/'

    imgs = os.listdir(pred_path)
    b = len(imgs)
    c = 0
    xdata = []
    ydata = []
    # tpr_each_image = np.zeros(a)
    # fpr_each_image = np.zeros(a)

    for name in imgs:

        imgl = cv2.imread(lab_path + '/' + name, -1)
        # print('imgl',imgl)
        imgl = np.array(imgl)


        image_map = cv2.imread(pred_path + name, -1)
        # print('image_map',image_map)
        image_map = np.array(image_map)

        tpr, fpr = calculate_score_threshold(image_map, imgl)
        c += 1
        # print('已经计算图像：' + str(c) + ',剩余数目：' + str(b - c))

        xdata.append(fpr)
        ydata.append(tpr)

    # averaging
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    FPR_values[n] = np.mean(xdata)
    TPR_values[n] = np.mean(ydata)
    n += 1

    c = n
    print('已经计算threshold：' + str(c) + ',剩余数目：' + str(101 - c))

    # Saving the ROC values

    dataframe = pd.DataFrame({'FPR': FPR_values, 'TPR': TPR_values})
    dataframe.to_csv("cloudunet_v2roc.csv", index=False, sep=',')

x = list(FPR_values)
y = list(TPR_values)


