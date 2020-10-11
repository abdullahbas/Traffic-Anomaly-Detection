#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:43:30 2020

@author: abas
"""

from matplotlib import pyplot as plt
from PIL import Image,ImageDraw
from skimage import measure
import  pandas as pd
import imutils
from matplotlib import patches
import cv2
import numpy as np
import yoloDet

class Perspective():
    def __init__(self,bw,frame,seg=5):
       
        self.seg=seg
        
        self.result,self.index1,self.result2,self.index2=self.Calculate(bw, frame)
        
        
        self.detector=yoloDet.yoloDetect('cfg/yolov3.cfg','yolov3.weights')

    def Calculate(self,bw,frame):
        seg=self.seg
        print('Helloo mother')
        # Take the bw map from CUDA to cpu and actually squeeze it with different way of view
        bwC=np.array(bw[0,0,:,:].cpu())
        
        # Find the exact indices of bw map --- It is not used because it is actually the exact same with map hahahah
        indsBW=np.where(bwC==1)
        bwCf=np.zeros(bwC.shape)
        bwCf[indsBW[0],indsBW[1]]=1
        # These lines are completely useless but who knows may be one day !!
        
        
        
        # We are labeling bw map with skimage morphologic library measure
        labels=measure.label(bwC)
        # This is for reaching to properties of connected objects in bw map
        props=measure.regionprops(labels)
        # Show frame to plot some fancy things on it
        #plt.imshow(frame)
        
        props=[x for x in props if x.area>sum(sum(bwC))*0.15]
        
        # Coords are the coordinates of pixels, we can reach it via line 22 which was labeled as useless! 
        # but we prefered to reach it with that way.
        coords=(props[0].coords)
        coords=pd.DataFrame(coords,columns=('aa','bb'))
        # Find the range of coords and group them on x axis coordinates to arrange y-pixel coordinates
        result=coords.groupby('aa').agg({'bb' : ['max','min']})
        
        # same /\ but the second connected map. Relax at first there will be more connected objects than 2 but
        # at last it will converge to 2 objects. I guess.
        coords=(props[1].coords)
        coords=pd.DataFrame(coords,columns=('aa','bb'))
        result2=coords.groupby('aa').agg({'bb' : ['max','min']})
        
        
        
        # index is the x coordinates that was gathered by performing unique process
        index1=(result.index.values)
        result=np.array(result)
        # same but belongs to second connected object
        index2=(result2.index.values)
        result2=np.array(result2)
        
        # First of the fancy things which we told at line 33
        #plt.scatter(result[:,1],index1,color='r')
        #plt.scatter(result[:,0],index1,color='b')
        # And as you guessed fancy things of second object.--Points of the borders 
        #plt.scatter(result2[:,1],index2,color='g')
        #plt.scatter(result2[:,0],index2,color='y')
        
        
        #inds are the points beyond the centroids of the connected objects. It is for computational power gain
        #we will not zoom into image parts which are before centroid
        inds1=(index1)<props[0].centroid[0]
        inds2=index2<props[1].centroid[0]
        
        # we have to manipulate our index which was x coordinates --line 55
        index1=index1[inds1]
        index2=index2[inds2]
        
        result=result[inds1,:]
        result2=result2[inds2,:]
        
        # We have to determine zoom segment counts.
        index1=index1[0:-1:len(index1)//seg]
        result=result[0:-1:len(result)//seg]
        # By the way fellas I hope you understand result/result2 are the y-coordinates of the objects.
        
        
        
        index2=index2[0:-1:len(index2)//seg]
        result2=result2[0:-1:len(result2)//seg]
        
        return result,index1,result2,index2
    def Perspective2Class(self,frame):
        # Take the bw map from CUDA to cpu and actually squeeze it with different way of view
       
        
        
        
        
        #for idx,xp in enumerate(index1):
         #   plt.plot([result[idx,0],result[idx,1]],[xp]*2)
          #  plt.plot([result2[idx,0],result2[idx,1]],[index2[idx]]*2)
        
        result=self.result
        index1=self.index1
        result2=self.result2
        index2=self.index2
        
        imgcrop=np.zeros(frame.shape)
        cropped=[]
        withBoundingBox=True
        for i in range(len(index1)-1):
            
            #rect=patches.Rectangle([index1[i],result[i,0]],result[i,0]-result[i,1],index1[i]-index1[i+1],-75,fill=True)
            imgcrop=np.zeros(frame.shape)
        
            roi=np.array([[result[i,0],index1[i]],[result[i,1],index1[i]],
                 [result[i+1,1],index1[i+1]],[result[i+1,0],index1[i+1]]])
            roi=np.round(roi*1).astype('int')
            rect=cv2.boundingRect(roi)
            x,y,w,h=rect
            if withBoundingBox:
                
                cv2.fillConvexPoly(imgcrop, np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]), (255,255,255))
            else:
                
                cv2.fillConvexPoly(imgcrop, roi, (255,255,255))
            
            imgcrop=imgcrop>10
            imgcrop=imgcrop*frame
            imgcrop=imgcrop[y:y+h,x:x+w]
            imgcrop=cv2.resize(imgcrop,tuple(frame.shape[0:2][::-1]))
            #orig,out,detections=detector.detect(imgcrop,confidence=0.01,nms_thesh=0.1)
            #ax1=plt.figure()
            #plt.imshow(orig)
            #imgcrop=imutils.rotate_bound(imgcrop, 75)
            
            cropped.append(imgcrop)
        
        
        
        
        imgcrop=np.zeros(frame.shape)
        cropped2=[]
        for i in range(len(index2)-1):
            
            #rect=patches.Rectangle([index1[i],result[i,0]],result[i,0]-result[i,1],index1[i]-index1[i+1],-75,fill=True)
            imgcrop=np.zeros(frame.shape)
        
            roi=np.array([[result2[i,0],index2[i]],[result2[i,1],index2[i]],
                 [result2[i+1,1],index2[i+1]],[result2[i+1,0],index2[i+1]]])
            
            rect=cv2.boundingRect(roi)
            x,y,w,h=rect
            if withBoundingBox:
                
                cv2.fillConvexPoly(imgcrop, np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]), (255,255,255))
            else:
                
                cv2.fillConvexPoly(imgcrop, roi, (255,255,255))    
            imgcrop=imgcrop>10
            imgcrop=imgcrop*frame
            imgcrop=imgcrop[y:y+h,x:x+w]
            imgcrop=cv2.resize(imgcrop,tuple(frame.shape[0:2][::-1]))
            #orig,out,detections=detector.detect(imgcrop,confidence=0.01,nms_thesh=0.1)
        
            #imgcrop=imutils.rotate_bound(imgcrop, 75)
            #ax1=plt.figure()
            #plt.imshow(orig)
            
            cropped2.append(imgcrop)
            
            #plt.draw()
        return cropped,cropped2
