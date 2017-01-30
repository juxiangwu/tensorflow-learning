# -*- coding: utf-8 -*-
"""
图像金字塔
图像金字塔是以多分辨率来解释图像的一种结构。
1987年，在一种全新而有效的信号处理与分析方法，即多分辨率理论中，小波首次作为分析基础出现了。
多分辨率理论将多种学科的技术有效地统一在一起，如信号处理的子带编码、数字语音识别的积分镜像过滤以及金字塔图像处理。
正如其名字所表达的，多分辨率理论与多种分辨率下的信号（或图像）表示和分析有关。
其优势很明显，某种分辨率下无法发现的特性在另一种分辨率下将很容易被发现。
参考:http://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E9%87%91%E5%AD%97%E5%A1%94
"""
import cv2
import numpy as np,sys

A = cv2.imread('../../../datas/images/apple.jpg')
B = cv2.imread('../../../datas/images/pear.jpg')
print(A.shape)
print(B.shape)
# 生成高斯金字塔
G = A.copy()
gpA = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpA.append(G)
    print(G.shape)

G = B.copy()
gpB = [G]
for i in range(5):
    G = cv2.pyrDown(G)
    gpB.append(G)
    #print(G.shape)

# 产生Laplacian金字塔
lpA = [gpA[5]]
for i in range(5,0,-1):
    print(i)
    print(gpA[i].shape)
    print(gpA[i - 1].shape)
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# 合并
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# 重新构建图像
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# 连接
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imshow("LS",ls_)
cv2.imshow("Real",real)

cv2.waitKey()
cv2.destroyAllWindows()