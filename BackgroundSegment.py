#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:30:29 2020

@author: abas
"""

import numpy as np
import cv2
import medianTorch
import torch
import yoloDet

    
class BGBW():

    def __init__(self):
        
        self.fgbg=cv2.createBackgroundSubtractorMOG2()
        self.mask=torch.zeros(1,1,410,800).cuda()
        self.bg1=torch.zeros(1,3,410,800).cuda()

        self.detector=yoloDet.yoloDetect('cfg/yolov3.cfg','yolov3.weights')
        
        
        
    def bgbw(self,frame,count):
        detector=self.detector        
        fgbg=self.fgbg
        mask=self.mask
        bg1=self.bg1
        fgmask = fgbg.apply(frame)
        frame2=frame.copy()
        orig,out,detections=detector.detect(frame2,confidence=0.3,nms_thesh=0.1)
        cv2.imshow('fgmask',orig)
        cv2.imshow('frame',fgmask)
        mod=medianTorch.MedianPool2d(7,padding=3)
        mask2=mod(torch.tensor(fgmask).unsqueeze(0).unsqueeze(0).cuda())
        
        
        maskRGB=torch.cat((mask2,mask2,mask2),dim=1);
        frameRGB=torch.tensor(frame).transpose(0,2).transpose(1,2).unsqueeze(0).cuda()
        
        
        mask2Inv=(mask2.max()-mask2).type(torch.float64)
        mask2Bw=mask2Inv>mask2Inv.mean()
        bg=mask2Bw*frameRGB
        bg1+=bg
        
        
    
        count+=1
        
        mask+=mask2
        bw = mask> mask.mean()
        
        bwS=bw.squeeze(0).squeeze(0)
        bgS=bg1.transpose(1,3).transpose(1,2).squeeze(0)/count
        cv2.imshow('BW',np.array(bwS.cpu()).astype('float64'))
        
        cv2.imshow('BG',np.array(bgS.cpu())/255)
        
        
        #orig,out,detections=detector.detect(frame)
        
        #cv2.imshow('Detections',orig)
        
        
        
        return bw,bgS.cpu()/255,detections


