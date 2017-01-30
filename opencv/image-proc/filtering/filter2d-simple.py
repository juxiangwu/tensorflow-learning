# -*- coding: utf-8 -*-
import  cv2 as cv
import numpy as np
kernel = np.array([[1, 1, 1],
                   [1, -8, 1],
                  [1, 1, 1]])

kernel2 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
[0.04, 0.04, 0.04, 0.04, 0.04],
[0.04, 0.04, 0.04, 0.04, 0.04],
[0.04, 0.04, 0.04, 0.04, 0.04],
[0.04, 0.04, 0.04, 0.04, 0.04]])

kernel3 = np.array([[-2, -1, 0],
[-1, 1, 1],
[ 0, 1, 2]])

img = cv.imread('../../../datas/images/fish.jpg')

dst = cv.filter2D(img,-1,kernel)
dst2 = cv.filter2D(img,-1,kernel2)
dst3 = cv.filter2D(img,-1,kernel3)

cv.imshow("res",dst)
cv.imshow("res2",dst2)
cv.imshow("res3",dst3)

cv.waitKey()
cv.destroyAllWindows()