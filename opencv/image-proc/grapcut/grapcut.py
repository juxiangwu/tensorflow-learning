# -*- coding: utf-8 -*-
"""
图像分割
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../../../datas/images/building.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
# 背景模型
bgdModel = np.zeros((1,65),np.float64)
# 前景模型
fgdModel = np.zeros((1,65),np.float64)

rect = (50,50,450,290)
# 使用grabCut算法
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img),plt.colorbar()

plt.show()