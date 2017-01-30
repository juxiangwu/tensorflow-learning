# -*- coding: utf-8 -*-
"""
Meanshift算法目标跟踪
Meanshift算法可以参考：http://blog.csdn.net/carson2005/article/details/7337432
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# 读取摄像头第一帧图像
ret, frame = cap.read()
while True:
    ret, frame = cap.read()
    if ret == True:
        break
# 初始化位置窗口
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# 设置所要跟踪的ROI
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        else:
           # cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
