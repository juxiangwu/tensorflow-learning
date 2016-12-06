# -*- coding: utf-8 -*-
# 图像几何变换

import cv2
import numpy as np

# 大小变换
img = cv2.imread('../../../datas/images/fish.jpg')
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
# 第二种大小变换方法
height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

# 平移变换
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
rows,cols = gray.shape
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(gray,M,(cols,rows))
cv2.imshow("image",dst)

# 旋转
M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = dst = cv2.warpAffine(gray,M,(cols,rows))
cv2.imshow("rotate",dst)

# 仿射变换
rows,cols,ch = img.shape
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow("image",img)
cv2.imshow("affine transform",dst)

# 透视变换
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

cv2.imshow("perspective transform",dst)

cv2.waitKey()
cv2.destroyAllWindows()
