#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 12:37:53 2020

@author: abas
"""

from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img,classes,colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if cls  in [2,3,5,6,7]:
    
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img




class yoloDetect():
    
    def __init__(self,cfgfile,weightsfile):
        self.cfg=cfgfile
        self.weightsfile=weightsfile
        model = Darknet(self.cfg)
        model.load_weights(self.weightsfile)
        self.model=model
    
    def detect(self,frame,confidence=0.4,nms_thesh=0.5):
        start = 0
    
        CUDA = torch.cuda.is_available()
    
        num_classes = 80    
        bbox_attrs = 5 + num_classes
    
        
        model=self.model
        
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32
    
        if CUDA:
            model.cuda()
            
    
        
    
        
        
        outpt=[]    
        frames = 0
            
        
        
        img, orig_im, dim = prep_image(frame, inp_dim)
                
        im_dim = torch.FloatTensor(dim).repeat(1,2)                        
        if CUDA:
                    im_dim = im_dim.cuda()
                    img = img.cuda()
                
        with torch.no_grad():   
                    output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        outpt.append(np.array(output.cpu()))
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor
    
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
        
        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))
        
        list(map(lambda x: write(x, orig_im,classes,colors), output))
        
        
        return orig_im,output,outpt


    
    

