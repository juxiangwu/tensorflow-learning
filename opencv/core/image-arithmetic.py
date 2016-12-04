#-*- coding: utf-8 -*-
# 图像操作
import cv2 as cv

img1 = cv.imread('../../datas/images/fish.jpg')
img2 = cv.imread('../../datas/images/fish2.jpg')

cv.imshow("img1",img1)
cv.imshow("img2",img2)

# 图像简单混合
dist1 = cv.add(img1,img2)
cv.imshow("dist1",dist1)

alpha = 0.75
beta = 1.0 - alpha
gamma = 2.5

dist2 = cv.addWeighted(img1,alpha,img2,beta,gamma)

cv.imshow("dist2",dist2)

# 获取图像ROI
rows,cols,channels = img1.shape
roi = img1[10:rows / 2,10:cols / 2]
cv.imshow("roi",roi)

# 转换颜色空间
gray = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
cv.imshow("gray",gray)

# 图像阈值操作
ret,mask = cv.threshold(gray,10,255,cv.THRESH_BINARY)
cv.imshow("thresh-mask",mask)

mask_inv = cv.bitwise_not(mask)
cv.imshow("mask-inv",mask_inv)

# 简单分离背景
roi = img1[0:rows,0:cols]
img1_bg = cv.bitwise_and(roi,roi,mask,mask_inv)
cv.imshow('image-bg',img1_bg)

img2_fg = cv.bitwise_and(img2,img2,mask=mask)
cv.imshow("img2-fg",img2_fg)

# 通过ROI改变主图
dist = cv.add(img1_bg,img2_fg)
img1[0:rows,0:cols] = dist
cv.imshow('dist',img1)

cv.waitKey()
cv.destroyAllWindows()
