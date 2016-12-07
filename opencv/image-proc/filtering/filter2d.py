# -*- coding: utf-8 -*-
# 图像滤波
'''
图像处理也支持低通滤波(LPF)和高通滤波(HPF)处理
OpenCV提供filter2D函数对图像进行滤波处理
'''

import  cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv.imread('../../../datas/images/fish.jpg')

# 滤波Kernel,求平均值
kernel = np.ones((5,5),np.float32) / 25

# 均值滤波
# 执行滤波处理,图像将变得模糊
dst = cv.filter2D(img,-1,kernel)
# dist = cv.blur(img,(5,5))

# # 显示结果
# plt.subplot(121)
# plt.imshow(img)
# plt.title('Original')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(dst)
# plt.title('Averaging')
# plt.xticks([])
# plt.yticks([])

#plt.show()

cv.imshow("image",img)
cv.imshow("Filter-Avg",dst)

# 高斯滤波
dst = cv.GaussianBlur(img,(5,5),0)
cv.imshow("Gaussian Blur",dst)

# 中值滤波
dst = cv.medianBlur(img,5)
cv.imshow("Median Blur",dst)

# 双边滤波,可以用来做一些简单的美颜处理
dst = cv.bilateralFilter(img,9,75,75)
cv.imshow("Bilateral Filter",dst)

cv.waitKey()
cv.destroyAllWindows()



