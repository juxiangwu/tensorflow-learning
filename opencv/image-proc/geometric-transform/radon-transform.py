# -*- coding: utf-8 -*-
"""
Radon变换
"""
import cv2
import numpy as np

img = cv2.imread('../../../datas/images/fish.jpg',0)
rows,cols = img.shape[:2]
print("rows = ",rows)
print("cols = ",cols)
dst = img[:]#np.zeros((int(rows),int(cols),1),np.float32)

angle = int(360)
radon_image = np.zeros((int(rows),int(angle),1),np.float32)

center = int(rows / 2)

shift0 = [1.0,0.0,float(-center),0.0,1.0,float(-center),0.0,0.0,1.0]
shift1 = [1.0,0.0,float(center),0.0,1.0,float(center),0.0,0.0,1.0]

m0 = np.array(shift0).reshape(3,3)
m1 = np.array(shift1).reshape(3,3)
theta = [0] *360

for t in range(360):
    theta[t] = t * np.pi / angle

    R = [np.cos(theta[t]),np.sin(theta[t]),0,
        -np.sin(theta[t]),np.cos(theta[t]),0,
        0,0,1]
    mR = np.array(R).reshape(3,3)

    rotation = m1 * mR * m0

    rotated = cv2.warpPerspective(dst,rotation,(rows,cols),cv2.WARP_INVERSE_MAP)
    rrows,rcols = rotated.shape[:2]

    for j in range(rcols):
        p1 = radon_image[j]
        for i in range(rrows):
            p2 = rotated[i]
            p1[t] += p2[j]
result = cv2.normalize(radon_image,0,1,cv2.NORM_MINMAX)
cv2.imshow("result",radon_image)
cv2.waitKey()


