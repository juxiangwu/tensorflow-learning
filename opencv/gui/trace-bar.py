# -*- coding: utf-8 -*-
# 处理滚动条事件

import cv2 as cv
import numpy as np

def on_trace_bar_changed(args):
    pass

img = np.zeros((512,512,3),np.uint8)

cv.namedWindow("image")

# 创建滚动条
cv.createTrackbar("R","image",0,255,on_trace_bar_changed)
cv.createTrackbar("G","image",0,255,on_trace_bar_changed)
cv.createTrackbar("B","image",0,255,on_trace_bar_changed)

while True:
    cv.imshow("image",img)
    k = cv.waitKey(10) & 0xFF
    if k == 27:
        break
    # 获取滚动条的值
    r = cv.getTrackbarPos("R","image")
    g = cv.getTrackbarPos("G","image")
    b = cv.getTrackbarPos("B","image")

    img[:] = [b,g,r]