# -*- coding: utf-8 -*-

"""
图像轮廓属性
"""

import numpy as np
import cv2

img = cv2.imread('../../../datas/images/building.jpg')
img2 = cv2.imread('../../../datas/images/building.jpg')
img3 = cv2.imread('../../../datas/images/building.jpg')
img4 = cv2.imread('../../../datas/images/building.jpg')
img5 = cv2.imread('../../../datas/images/building.jpg')
img6 = cv2.imread('../../../datas/images/building.jpg')

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 阈值化
ret,thresh = cv2.threshold(imgray,127,255,0)
# 查找轮廓
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (0,255,0), 1)
cv2.imshow("contours",img)


rows, cols = img.shape[:2]

size = len(contours)
for i in range(size):
    # 计算monents
    M = cv2.moments(contours[i])
    if M['m10'] != 0.0 and M['m00'] !=0.0:
        cx = int(M['m10']/M['m00'])
        print("cx = ", cx)
    if M['m01'] != 0.0 and M['m00'] != 0.0:
        cy = int(M['m01']/M['m00'])
        print("cy = ",cy)
    # 面积
    area = cv2.contourArea(contours[i])
    print("area = ",area)

    # 周长
    perimeter = cv2.arcLength(contours[i],True)
    print("perimeter = ",perimeter)

    # Fitting a line
    [vx,vy,x,y] = cv2.fitLine(contours[i],cv2.DIST_L2,0,0.01,0.01)
    if vx >= 0 and vy >= 0:
        lefty = int((-x * vy) / vx + y)
        righty = int(((cols - x) * vy / vx) + vy)
        print("vx = ",vx)
        print("vy = ",vy)
        print("lefty = ",lefty)
        print("righty = ",righty)
        cv2.line(img2,(cols - 1,righty),(0,lefty),(0,255,0),1)

    # Fitting an Ellipse
    if area > 0 and len(contours[i]) > 5:
        ellipse = cv2.fitEllipse(contours[i])
        cv2.ellipse(img3, ellipse, (0, 255, 0), 1)

    # Minimum Enclosing Circle
    (x,y),radius = cv2.minEnclosingCircle(contours[i])
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img4,center,radius,(0,255,0),1)

    # Rotated Rectangle
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img5, [box], 0, (0, 0, 255), 1)

    # Bounding Rectangle
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(img6, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Checking Convexity
    k = cv2.isContourConvex(contours[i])
    print("Is Contour Convex = ",k)

    # Convex Hull
    hull = cv2.convexHull(contours[i])
    print('Convex hull',hull)

cv2.imshow("Fitting Lines",img2)
cv2.imshow("Fitting Ecllipses",img3)
cv2.imshow("Minimum Enclosing Circle",img4)
cv2.imshow("Rotated Rectangle",img5)
cv2.imshow("Bounding Rectangle",img6)

cv2.waitKey()
cv2.destroyAllWindows()