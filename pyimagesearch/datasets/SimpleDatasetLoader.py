#!/usr/bin/env python
__author__="Shashikiran Alloli"
__version__="0.1"

import os
import cv2
import numpy as np

class simpledatasetloader:
    def __init__(self, preprocessors=None):
        if preprocessors is None:
            self.preprocessors=[]
        else:
            self.preprocessors=preprocessors
    
    def load(self, images, verbose=-1):
        data=[]
        labels=[]

        for i, imagepath in enumerate(images):
            label=imagepath.split(os.path.sep)[-2]
            image=cv2.imread(imagepath)

            for preprocessor in preprocessors:
                image=preprocessor.fit_transform(image)

            data.append(image)
            labels.append(label)

            if(verbose>0 and (i+1)%verbose==0):
                print("Processing {} file...".format(i+1))
        
        return data, labels
        
