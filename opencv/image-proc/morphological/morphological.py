# -*- coding: utf-8 -*-
"""
图像数学形态操作
"""

import cv2
import  numpy as np

img = cv2.imread('../../../datas/images/building.jpg')
kernel = np.ones((5,5),np.uint8)

# 腐蚀
erosion = cv2.erode(img,kernel,iterations=1)
cv2.imshow("Erosion",erosion)

# 膨胀
dialation = cv2.dilate(img,kernel,iterations=1)
cv2.imshow("Dialation",dialation)

# 开操作
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow("Opening",opening)

# 闭操作
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=1)
cv2.imshow("Closing",closing)

# 形态梯度
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("Gradient",gradient)

# Top Hat
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Top Hat",tophat)

# Black Hat
blackhat = tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("Black Hat",blackhat)

# 结构化元素
# 矩形内核
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
print("Rectangular Kernel = ")
print(rect_kernel)

# 椭圆内核
ecllipital_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
print("Ellipital Kernel = ")
print(ecllipital_kernel)

# 十字形内核
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
print('Cross-shaped Kernel = ')
print(cross_kernel)


cv2.waitKey()
cv2.destroyAllWindows()



