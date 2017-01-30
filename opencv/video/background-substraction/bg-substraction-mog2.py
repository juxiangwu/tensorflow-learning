# -*- coding: utf-8 -*-
"""
视频背景消除
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    fgmask = fgbg.apply(frame)
    cv2.imshow("frame",fgmask)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()