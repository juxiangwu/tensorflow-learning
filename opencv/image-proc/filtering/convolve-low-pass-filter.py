# -*- coding: utf-8 -*-
'''
低通卷积滤波
'''

import cv2 as cv
import  numpy as np
from scipy import ndimage

img = cv.imread("../../../datas/images/fish.jpg",0)