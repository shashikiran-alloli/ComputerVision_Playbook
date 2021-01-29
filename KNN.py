#!/usr/bin/env python
__author__="Shashikiran Alloli"
__version__="0.1"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import SimplePreprocessor
from imutils import paths
import os
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dataset", required=True, help="Path containing images")
argparser.add_argument("-k", "--neighbors", type=int, default=1, help="k nearest neighbors")
argparser.add_argument("-j", "--jobs", type=int, default=-1)
args=vars(argparser.parse_args())

imagepaths=list(paths.list_images(args['dataset']))
sp=SimplePreprocessor.simplepreprocessor(32,32)
sdl=SimpleDatasetLoader.simpledatasetloader([sp])
le=LabelEncoder()

data, labels = sdl.load(imagepaths, 5)

data.reshape((data.shape[0], 3072))
labels = le.fit_transform(labels)

trainX, trainY, testX, testY = train_test_split(data, train_size=0.25, random_state=42)

knn=KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
knn.fit(trainX, trainY)
classification_report(testY, knn.predict(testX), target_names=le.classes_)

