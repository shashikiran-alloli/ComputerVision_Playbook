__author__ = "Shashikiran Alloli"
__version__ = "0.1"

#Library imports
from pyimagesearch.nn import NeuralNetwork
import numpy as np

#Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

#Training
nn = NeuralNetwork(layers=[2,2,1], alpha=0.1)
nn.fit(X, y, epochs=20000, display=100)

#Predict
for (x, target) in zip(X,y):
    pred = nn.predict(x)
    out = 1 if pred>0.5 else 0
    print("[Output] Input: {} and Prediction: {} and Actual output: {}".format(x, pred, out))





