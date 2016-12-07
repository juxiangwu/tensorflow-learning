# -*- coding: utf-8 -*-
"""
计算彩色图像各通道的直方图及图像区域直方图
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../../../../datas/images/fish.jpg')
color = ('b','g','r')
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

# 使用Mask计算某区域直方图
img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
mask = np.zeros(img_gray.shape[:2],np.uint8)
mask[100:200,100:200] = 255
masked_img = cv.bitwise_and(img_gray,img_gray,mask = mask)
hist_full = cv.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])

plt.figure()
plt.subplot(221)
plt.imshow(img_gray,'gray')
plt.subplot(222)
plt.imshow(mask,'gray')
plt.subplot(223)
plt.imshow(masked_img,'gray')
plt.subplot(224)
plt.plot(hist_full)
plt.plot(hist_mask)
plt.xlim([0,256])


plt.show()