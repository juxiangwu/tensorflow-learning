# -*- coding: utf-8 -*-
"""
图像的直方图计算
"""
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("../../../../datas/images/fish.jpg",0)
hist = cv.calcHist([img],[0],None,[256],[0,256])
plt.subplot(121)
plt.imshow(img,'gray')
plt.xticks([])
plt.yticks([])
plt.title("Original")
plt.subplot(122)
plt.hist(img.ravel(),256,[0,256])
plt.show()
