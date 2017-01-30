# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../../../datas/images/fish.jpg',0)

# 初始化FAST对象
fast = cv2.FastFeatureDetector_create()

# 查找和绘制关键点
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))

cv2.imshow("fast-true",img2)
cv2.imshow("image",img)

# 禁用nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2.imshow("fast-false",img3)

cv2.waitKey()
cv2.destroyAllWindows()