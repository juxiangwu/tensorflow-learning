# -*- coding: utf-8 -*-
# 图像Canny边缘检测
"""
    Canny算法是John F. Canny in 1986发明的一个多级边缘检测算法。实现步骤如下：
    1、应用高斯滤波来平滑图像，目的是去除噪声
    2、找寻图像的强度梯度（intensity gradients）
    3、应用非最大抑制（non-maximum suppression）技术来消除边误检（本来不是但检测出来是）
    4、应用双阈值的方法来决定可能的（潜在的）边界
    5、利用滞后技术来跟踪边界
    参考：http://baike.baidu.com/item/canny%E7%AE%97%E6%B3%95
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 读取灰度图像
img = cv.imread("../../../datas/images/fish.jpg",0)

# 执行边缘检测
edges = cv.Canny(img,100,200)

plt.subplot(121)
# 显示灰度图像
plt.imshow(img,cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(edges,cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title("Edge detect result")

plt.show()

# cv.imshow("Image",img)
# cv.imshow("Edge-Canny",edges)
#
# cv.waitKey()
# cv.destroyAllWindows()

