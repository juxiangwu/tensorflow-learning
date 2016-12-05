#-*- coding: utf-8 -*-
# 读取，保存，显示图片
import cv2 as cv

# 读取为灰度图片
img = cv.imread("../../datas/images/fish.jpg",0)

# 保存图片
cv.imwrite("../../datas/images/fish-gray.jpg",img=img)

# 显示图片
cv.imshow("img-gray",img)
'''
# 通过Matplotlib显示图片
#import  matplotlib.pyplot as plt
#plt.imshow(img,cmap='gray',interpolation='bicubic')
#plt.xticks([])
#plt.yticks([])
#plt.show()
#cv.waitKey()
#cv.destroyAllWindows()
'''
cv.waitKey()
cv.destroyAllWindows()
'''
# 等待事件
# key = cv.waitKey(10)
# if key == 27:# ESC
# 销毁所有窗口
#cv.destroyAllWindows()
'''
