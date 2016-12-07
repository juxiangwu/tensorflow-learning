# -*- coding: utf-8 -*-
"""
图像直方图反向投影简单实现
如果一幅图像的区域中显示的是一种结构纹理或者一个独特的物体，那么这个区域的直方图可以看作一个概率函数，
它给的是某个像素属于该纹理或物体的概率。
所谓反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找测试图像中存在的该特征。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# roi is the object or region of object we need to find
roi = cv2.imread('../../../../datas/images/fish.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

# target is the image we search in
target = cv2.imread('../../../../datas/images/fish-target.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )

R = M/(I+1)
#p rint R.max()
# cv2.normalize(prob,prob,0,255,cv2.NORM_MINMAX,0)

h,s,v = cv2.split(hsvt)
B = R[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)

ret,thresh = cv2.threshold(B,50,255,0)
res = cv2.bitwise_and(target,target,mask = thresh)
cv2.imshow('nice',res)
cv2.imshow('img',target)
res = np.vstack((target,cv2.merge((B,B,B)),res))
#cv2.imwrite('thresh.png',thresh)
#cv2.imwrite('output.png',res)
cv2.imshow("result",res)

cv2.waitKey(0)
cv2.destroyAllWindows()