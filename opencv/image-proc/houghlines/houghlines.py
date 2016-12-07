# -*- coding: utf-8 -*-
"""
hough变换检测直线
"""
import cv2
import numpy as np

img = cv2.imread('../../../datas/images/building.jpg')
cv2.imshow("img",img)
print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
# 检测边缘
edges = cv2.Canny(gray,80,120,apertureSize = 5)
cv2.imshow("edges",edges)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
if len(lines) > 0:
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

#cv2.imwrite('houghlines3.jpg',img)
cv2.imshow("result",img)
cv2.waitKey()
cv2.destroyAllWindows()