#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:37:21 2020

@author: abas
"""
import time

import cv2
import yoloDet
from perspective import Perspective
from BackgroundSegment import BGBW

detector=yoloDet.yoloDetect('cfg/yolov3.cfg','yolov3.weights')
bck=BGBW()
cap = cv2.VideoCapture('2.mp4')
count=1
frames=1
countPers=1
detections=[]
start=time.time()
while(1):
    ret, frame = cap.read()

    bw,bg,dt=bck.bgbw(frame,count)
    detections.append(dt)
    if count>999:
        
        if count%1000==0:
            pers=Perspective(bw,frame)
            frames=1
            start=time.time()
        cr1,cr2=pers.Perspective2Class(frame)
        zoomFrame1=cv2.hconcat((cr1[0],cr1[1]))
        zoomFrame2=cv2.hconcat((cr1[2],cr1[3]))
        zoomFrame1=cv2.vconcat((zoomFrame1,zoomFrame2))

        detector.detect(zoomFrame1,confidence=0.1,nms_thesh=0.1)
        cv2.imshow('Zoomed',zoomFrame1)
        
        zoomFrame2=cv2.hconcat((cr2[0],cr2[1]))
        zoomFrame22=cv2.hconcat((cr2[2],cr2[3]))
        zoomFrame2=cv2.vconcat((zoomFrame2,zoomFrame22))

        detector.detect(zoomFrame2,confidence=0.1,nms_thesh=0.1)
        cv2.imshow('Zoomed2',zoomFrame2)
        
        
        
    
    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))        
        
    count+=1
    frames+=1

    k = cv2.waitKey(1) 
    if k  & 0xff== ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()


