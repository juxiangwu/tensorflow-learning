# -*- coding: utf-8 -*-
"""
图像模板匹配
模板匹配是在图像中寻找目标的方法之一
模板匹配的工作方式
 模板匹配的工作方式跟直方图的反向投影基本一样，大致过程是这样的：通过在输入图像上滑动图像块对实际的图像块和输入图像进行匹配。
 假设我们有一张100x100的输入图像，有一张10x10的模板图像，查找的过程是这样的：
 （1）从输入图像的左上角(0,0)开始，切割一块(0,0)至(10,10)的临时图像；
 （2）用临时图像和模板图像进行对比，对比结果记为c；
 （3）对比结果c，就是结果图像(0,0)处的像素值；
 （4）切割输入图像从(0,1)至(10,11)的临时图像，对比，并记录到结果图像；
 （5）重复（1）～（4）步直到输入图像的右下角。
模板匹配的匹配方式
    在OpenCv和EmguCv中支持以下6种对比方式：
    CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
    CV_TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
    CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
    CV_TM_SQDIFF_NORMED 归一化平方差匹配法
    CV_TM_CCORR_NORMED 归一化相关匹配法
    CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
参考：http://www.cnblogs.com/xrwang/archive/2010/02/05/MatchTemplate.html
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../../datas/images/apple.jpg',0)
img2 = img.copy()
template = cv2.imread('../../../datas/images/apple-template.jpg',0)
w, h = template.shape[::-1]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    plt.figure()
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

plt.show()