# -*- coding: utf-8 -*-
"""
图像直方图反向投影
如果一幅图像的区域中显示的是一种结构纹理或者一个独特的物体，那么这个区域的直方图可以看作一个概率函数，
它给的是某个像素属于该纹理或物体的概率。
所谓反向投影就是首先计算某一特征的直方图模型，然后使用模型去寻找测试图像中存在的该特征。
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../../../datas/images/fish-target.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

target = cv2.imread('../../../../datas/images/fish.jpg')
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
dst2 = dst.copy()
ret,thresh = cv2.threshold(dst,50,255,0)
thresh = cv2.merge((dst,dst,dst))
dst2 = cv2.merge((dst,dst,dst))
res = cv2.bitwise_and(target,thresh)

res = np.vstack((target,dst2,res))

#cv2.imwrite('res.jpg',res)
cv2.imshow("result",res)
cv2.waitKey()
cv2.destroyAllWindows()