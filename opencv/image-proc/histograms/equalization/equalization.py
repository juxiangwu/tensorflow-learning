# -*- coding: utf-8 -*-
"""
图像直方图均衡化
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../../../datas/images/fish.jpg',0)
# 计算处理前的直方图
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.figure()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

# 均衡化处理
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

plt.figure()
plt.subplot(121)
plt.imshow(img,'gray')
plt.subplot(122)
plt.imshow(img2,'gray')

plt.figure()
# 处理后直方图
hist,bins = np.histogram(img2.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

# 显示处理后直方图
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

# 使用OpenCV提供的函数
equ = cv2.equalizeHist(img)
plt.figure()
plt.subplot(121)
plt.imshow(img,'gray')
plt.subplot(122)
plt.imshow(equ,'gray')

plt.show()
