import numpy as np
import cv2
import medianTorch
import torch
from scipy.fft import fft,ifft
from skimage import morphology
import perspective
from matplotlib import pyplot as plt
import yoloDet



cap = cv2.VideoCapture('2.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
mask=torch.zeros(1,1,410,800).cuda()
bg1=torch.zeros(1,3,410,800).cuda()
count=1
countPers=1
detector=yoloDet.yoloDetect('cfg/yolov3.cfg','yolov3.weights')
while(1):
    ret, frame = cap.read()
    
        
    
        
        
    
    fgmask = fgbg.apply(frame)

    
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)
    mod=medianTorch.MedianPool2d(7,padding=3)
    mask2=mod(torch.tensor(fgmask).unsqueeze(0).unsqueeze(0).cuda())
    
    
    maskRGB=torch.cat((mask2,mask2,mask2),dim=1);
    frameRGB=torch.tensor(frame).transpose(0,2).transpose(1,2).unsqueeze(0).cuda()
    
    
    mask2Inv=(mask2.max()-mask2).type(torch.float64)
    mask2Bw=mask2Inv>mask2Inv.mean()
    bg=mask2Bw*frameRGB
    print(bg.shape)
    bg1+=bg
    
    
    
    print(bg1.max())
    print(bg1.min())
    count+=1
    print(count)
    
    mask+=mask2
    bw = mask> mask.mean()
    
    bwS=bw.squeeze(0).squeeze(0)
    bgS=bg1.transpose(1,3).transpose(1,2).squeeze(0)/count
    print(bgS.shape)
    cv2.imshow('BW',np.array(bwS.cpu()).astype('float64'))

    cv2.imshow('BG',np.array(bgS.cpu())/255)
    

    #orig,out,detections=detector.detect(frame)

    #cv2.imshow('Detections',orig)
    '''
    try:
        
        pers=perspective.Perspective(bw,frame)
        props=pers.Perspective2Class()
        
        ax1=plt.subplot(3,1,1)
        plt.imshow(cropped[0])
        ax1.subplot(3,1,2)
        ax1.imshow(cropped[1])
        ax1.subplot(3,1,3)
        ax1.imshow(cropped[2])
        
        
    except:
        print('Waiting for bw')
    
    '''

    k = cv2.waitKey(1) 
    if k  & 0xff== ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()


