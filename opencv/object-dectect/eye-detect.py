# -*- coding: utf-8 -*-
"""
眼睛检测
"""
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../../datas/misc/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../../datas/misc/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        # 提取脸部区域
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h,x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey +eh),(0,255,0),2)
    cv2.imshow("Camera",frame)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()