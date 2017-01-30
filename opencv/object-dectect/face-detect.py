# -*- coding: utf-8 -*-
"""
脸部检测
"""
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../../datas/misc/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('../../datas/misc/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        px = int(x)
        py = int(y)
        pw = int(w)
        ph = int(h)
        cv2.rectangle(frame,(px,py),(px + pw,py + ph),(255,0,0),2)

    cv2.imshow("Camera",frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()