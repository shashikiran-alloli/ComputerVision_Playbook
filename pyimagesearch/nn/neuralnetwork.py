__author__ = "Shashikiran Alloli"
__version__ = "0.1"

#library imports
import numpy as np

#Constructing neural network
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []
        self.alpha = alpha
        self.layers = layers

        for layer in np.arange(0,len(layers)-2):
            w = np.random.randn(layers[layer]+1, layers[layer+1]+1)
            self.W.append(w/np.sqrt(layers[layer]))

        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):
        return "[INFO] Creating neural network with layers -> {}".format(self.layers)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
        return x*(1-x)


    def predict(self, X, bias=True):
        X = np.atleast_2d(X)
        if bias:
            X = np.c_[X, np.ones(X.shape[0])]
        for w in self.W:
            X = self.sigmoid(X.dot(w))
        return X


    def calculate_loss(self, X, y):
        loss=[]
        target = self.predict(X, bias=False)
        error = target - y

        return 0.5*np.sum(error**2)


    def fit(self, X, y, epochs=1000, display=10):
        X = np.atleast_2d(X)
        X = np.c_[X, np.ones(X.shape[0])]

        for epoch in np.arange(0, epochs):
            self.fit_partial(X, y)

            if epoch==0  or (epoch+1)%display==0:
                loss = self.calculate_loss(X,y)
                print("[INFO]: Epoch - {} and loss - {}".format(epoch, loss))


    def fit_partial(self, X, y):

        for (x,target) in zip(X,y):
            A = [np.atleast_2d(x)]

            for w in self.W:
                out = self.sigmoid(A[-1].dot(w))
                A.append(out)

            D = []
            error = A[-1] - target
            D.append(error*self.sigmoid_deriv(A[-1]))

            for layer in np.arange(len(self.layers)-2, 0, -1):
                #print("Shape of D[-1] ->",D[-1].shape)
                #print("Shape of W[layer] ->", self.W[layer].shape)
                #print("Shape of sigmoid_deriv(A[layer]) ->", self.sigmoid_deriv(A[layer]).shape)
                out = (D[-1].dot(self.W[layer].T))*self.sigmoid_deriv(A[layer])
                D.append(out)

            D = D[::-1]
            for (i,w) in enumerate(self.W):
                self.W[i] += -self.alpha*A[i].T.dot(D[i])




