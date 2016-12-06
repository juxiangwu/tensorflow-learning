# -*- coding: utf-8 -*-
# 颜色空间转换
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    # 获取图像阈值
    mask = cv.inRange(hsv,lower_blue,upper_blue)

    # 掩码操作
    cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow("frame",frame)
    cv.imshow("mask",mask)

    key = cv.waitKey(10) & 0xFF
    if key == 27:
        break

