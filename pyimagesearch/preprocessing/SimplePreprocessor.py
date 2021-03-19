#!/usr/bin/env python
__author__="Shashikiran Alloli"
__version__="0.1"
__all__=['simplepreprocessor']

import cv2

class simplepreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width=width
        self.height=height
        self.inter=inter

    def fit_transform(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

    
