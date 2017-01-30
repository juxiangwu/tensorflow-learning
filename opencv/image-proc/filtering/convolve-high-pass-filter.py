# -*- coding: utf-8 -*-
import cv2 as cv
import  numpy as np
from scipy import ndimage
'''
    高通卷积滤波
'''

# 读取灰度图像
img = cv.imread("../../../datas/images/fish.jpg",0)

kernel_3x3 = np.array([[-1, -1, -1],
[-1, 8, -1],
[-1, -1, -1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
[-1, 1, 2, 1, -1],
[-1, 2, 4, 2, -1],
[-1, 1, 2, 1, -1],
[-1, -1, -1, -1, -1]])

k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)
blurred = cv.GaussianBlur(img, (11,11), 0)
g_hpf = img - blurred
cv.imshow("3x3", k3)
cv.imshow("5x5", k5)
cv.imshow("g_hpf", g_hpf)
cv.waitKey()
cv.destroyAllWindows()
