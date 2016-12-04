#-*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

img = cv.imread('../../datas/images/cat1.png')
# 访问图像基本属性
print('image.shape = ',img.shape)#[rows,cols,channels]
print('image.size = ',img.size)
# 访问像素
px = img[100,100]
print(px)
# 访问像素中的蓝色分量值
px_blue = img[100,100,0]
print(px_blue)

# 快速访问和编辑像素值
px_r = img.item(10,10,2)
px_g = img.item(10,10,1)
px_b = img.item(10,10,0)
print(px_r,px_g,px_b)

# 设置像素分量值
img.itemset((10,10,2),100)
px_r = img.item(10,10,2)
print('after changed = ',px_r)

# 图像ROI
region = img[10:100,0:100]
cv.imshow("src",img)
cv.imshow("roi",region)

# 获取图像通道
b,g,r = cv.split(img)
res = cv.merge((r,g,b))# 不按BGR储存顺序
cv.imshow("merge",res)

b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]

cv.imshow("src-r",r)

imgcpy = img[:]

cv.imshow("img-copy",imgcpy)

# 设置指定通道
# 设置图像的红色通道值为0
imgcpy[:,:,2] = 0

cv.imshow("img-copy-changed",imgcpy)

cv.waitKey()
cv.destroyAllWindows()

