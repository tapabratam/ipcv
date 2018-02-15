# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:57:08 2018

@author: Rishav
"""

import cv2
import numpy as np
from collections import deque
cap=cv2.VideoCapture()
cap.open(0)
pts=deque(maxlen=32)
while cap.isOpened():
    #Reading the frame
    ret,frame=cap.read()
    frame=cv2.flip(frame,flipCode=1)# 1 signifies Flip along Y axis
    #Changing the colour space
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Creating the mask
    lower_red=np.array([160,100,100])
    upper_red=np.array([179,255,255])
    mask=cv2.inRange(hsv,lower_red,upper_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #Finding the Contour
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center=None
    
    if len(contours)>0:
        c = max(contours, key=cv2.contourArea)
        ((x,y),rad)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))
        if(rad>10):
            cv2.circle(frame, (int(x), int(y)), int(rad),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), cv2.LINE_AA)
    cv2.imshow("MASK",mask)
    cv2.imshow("FRAME",frame)
    k=(cv2.waitKey(30)&0xFF)
    if(k==ord('q')):
        pts=deque(maxlen=32)
    if(k==27):
        break
cv2.destroyAllWindows()
cap.release()
    
