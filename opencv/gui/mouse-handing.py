# -*- coding: utf-8 -*-
# 处理鼠标事件

import cv2 as cv
import numpy as np
img = np.zeros((512,512,3),np.uint8)
ix,iy = -1,-1
drawing = False
mode = True

# 鼠标回调函数
def on_mouse_action(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN: # 鼠标左键按下
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_LBUTTONDBLCLK: # 鼠标左键双击
        cv.circle(img,(x,y),100,(255,0,0),-1)

    elif event == cv.EVENT_MOUSEMOVE: #鼠标移动
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)


cv.namedWindow("image")
cv.setMouseCallback("image",on_mouse_action)

while True:
    cv.imshow("image",img)
    k = cv.waitKey(20) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv.destroyAllWindows()