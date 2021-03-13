__author__="Shashikiran Alloli"
__version__="0.1"

#Library Imports
from nn import Perceptron
import numpy as np

#Defining datasets
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

#Training
model = Perceptron(X.shape[1], alpha=0.1)
model.fit(X, y, epochs=20)

#Prediction
for (x, target) in zip(X, y):
    pred = model.predict(x)

    print("Input: {}, Ground-Truth: {}, Predicted: {}".format(x, target, pred))
