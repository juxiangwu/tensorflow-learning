# -*- coding: utf-8 -*-
"""
BRIEF算法图像特征检测
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
print(cv2.__version__)
img = cv2.imread('../../../datas/images/fish.jpg',0)

# 初始化FAST检测器
star = cv2.xfeatures2d.StarDetector_create()
# 初始化BRIEF提取器
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# 查找特征点
kp = star.detect(img,None)
# 使用BRIEF算法计算特征描述
kp, des = brief.compute(img, kp)

print(brief.descriptorSize())
print(des.shape)

cv2.imshow("DES",des)
cv2.waitKey()