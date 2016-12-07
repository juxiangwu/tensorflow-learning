# -*- coding: utf-8 -*-

"""
图像轮廓处理
"""

import numpy as np
import cv2

img = cv2.imread('../../../datas/images/building.jpg')
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 阈值化
ret,thresh = cv2.threshold(imgray,127,255,0)
# 查找轮廓
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (0,255,0), 1)
cv2.imshow("image",img)

cv2.waitKey()
cv2.destroyAllWindows()
