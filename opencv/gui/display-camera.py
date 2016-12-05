# -*- coding: utf-8 -*-
# 显示摄像头

import cv2 as cv

cap = cv.VideoCapture(0)
cv.namedWindow("Camera")
while True and cap.isOpened():
    ret,frame = cap.read()
    if ret == False: # 读取图像可能失败
        continue
    cv.imshow("Camera",frame)
    key = cv.waitKey(10) & 0xFF
    if key == 27:
        break

cv.destroyAllWindows()
