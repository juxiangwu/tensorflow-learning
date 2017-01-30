# -*- coding: utf-8 -*-
"""
通过图像金字塔方式缩小放大
"""

import cv2
src = cv2.imread('../../../datas/images/apple.jpg')
rows,cols = src.shape[:2]
# 缩小图片
res = cv2.pyrDown(src,dstsize=(int(rows / 2),int(cols / 2)))
cv2.imshow("src",src)
cv2.imshow("res",res)

# 放大图片
res = cv2.pyrUp(src,dstsize=(int(rows * 2),int(cols *2)))
cv2.imshow("res2",res)


print(src.shape)
print(res.shape)
cv2.waitKey()
cv2.destroyAllWindows()

