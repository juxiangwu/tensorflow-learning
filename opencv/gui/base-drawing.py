#-*- coding: utf-8 -*-
# OpenCV基础绘图

import cv2 as cv
import numpy as np

# 创建一张RGB图片，大小为512x512
img = np.zeros((512,512,3),np.uint8)

# 绘制直线,起点(0,0),终点(511,511)，颜色BGR(255,0,0)，线宽为3
cv.line(img,(0,0),(511,511),(255,0,0),3)

cv.imshow("img-line",img)

# 绘制矩形
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.imshow("img-rect",img)

# 绘制圆形
cv.circle(img,(447,63), 63, (0,0,255), -1)
cv.imshow("img-circle",img)

# 绘制椭圆
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
cv.imshow("img-ellipse",img)

# 绘制多边形
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img,[pts],True,(0,255,255))
cv.imshow("img-polylines",img)

# 添加文本
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)
cv.imshow("img-text",img)

cv.waitKey()
cv.destroyAllWindows()
