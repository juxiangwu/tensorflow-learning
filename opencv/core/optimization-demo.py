#-*- coding: utf-8 -*-
# 判断OpenCV是否已经打开优化功能
import numpy as np
import cv2 as cv

img = cv.imread('../../datas/images/fish.jpg')
# 时间开始
e1 = cv.getTickCount()

for i in range(5,49,2):
    img = cv.medianBlur(img,i)

# 时间结束
e2 = cv.getTickCount()
# 计算耗时
t = (e2 - e1) / cv.getTickFrequency()

print("operation time usage:",t) # operation time usage: 0.31588224549033933

# 判断Opencv是否已经使用优化功能
print(cv.useOptimized()) #True,表示已经启用

# 现在关闭优化功能
cv.setUseOptimized(False)
# 重新测试

# 时间开始
e1 = cv.getTickCount()

for i in range(5,49,2):
    img = cv.medianBlur(img,i)

# 时间结束
e2 = cv.getTickCount()
# 计算耗时
t = (e2 - e1) / cv.getTickFrequency()

print("operation time usage:",t) # operation time usage: 0.5663654670682662

# 重新打开优化功能
cv.setUseOptimized(True)
