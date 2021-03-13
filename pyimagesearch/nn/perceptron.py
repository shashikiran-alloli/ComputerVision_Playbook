__author__="Shashikiran Alloli"
__version__="0.1"

'''
Library Imports
'''
import numpy as np

'''
Perceptron class
'''
class Perceptron:
    #Initialize weight matrix and alpha
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N+1)/np.sqrt(N)
        self.alpha = alpha

    #Activate function - step
    def step(self, x):
        return 1 if x>0 else 0

    #Training the model
    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones((X.shape[0],1))]

        for epoch in np.arange(0, epochs):

            for (x, target) in zip(X,y):
                pred = self.step(np.dot(x, self.W))

                if pred != target:
                    error = pred - target
                    self.W += -self.alpha*error*x

    #Prediction
    def predict(self, X, bias=True):
        X = np.atleast_2d(X)

        if bias is True:
            X = np.c_[X, np.ones((X.shape[0],1))]

        return self.step(np.dot(X, self.W))
