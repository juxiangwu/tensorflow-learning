# -*- coding: utf-8 -*-
# 读取和保存视频

import cv2 as cv
import numpy as np

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

videoCapture = cv.VideoCapture('../../datas/videos/video.avi')

#获取码率及尺寸
fs = videoCapture.get(cv.CAP_PROP_FPS)
fourccD = decode_fourcc(videoCapture.get(cv.CAP_PROP_FOURCC))
fourcc = videoCapture.get(cv.CAP_PROP_FOURCC)
size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('fs = ',fs)
print('size = ',size)
print('fourccD = ',fourccD)
print('fourcc = ',fourcc)
fourcc = cv.VideoWriter_fourcc(*'XVID')
# I420-avi, MJPG-mp4
videoWriter = cv.VideoWriter(filename='../../datas/videos/video-out.avi',
                             fourcc=fourcc,
                             fps=int(fs), frameSize=size)
img_empty = np.zeros((512,512,3),np.uint8)
while videoCapture.isOpened():
    sucess,frame = videoCapture.read()
    if sucess == True:
        gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        videoWriter.write(frame)
        cv.imshow("video",frame)
        cv.imshow("video-gray",gray)
    else:
        print("read frame failed")
        cv.imshow("video",img_empty)
        #break
    key = cv.waitKey(10) & 0xFF
    if key == 27:
        break
cv.destroyAllWindows()
