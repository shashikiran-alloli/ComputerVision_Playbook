__author__ = "Shashikiran Alloli"
__version__ = "0.1"

#Library imports
from nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import numpy as np

#Data loading
#digits = datasets.load_digits()
#X = digits.data.astype("float")
data = np.genfromtxt("datasets/digit-recognizer/train.csv",dtype='float',delimiter=",",skip_header=1)
data_test = np.genfromtxt("datasets/digit-recognizer/test.csv",dtype='float',delimiter=",",skip_header=1)
X, y = data[:,1:], data[:,0]
y = y.astype('int')
print("Shape of data ->",X.shape)
X = (X - X.min())/(X.max() - X.min())
data_test = (data_test - data_test.min())/(data_test.max() - data_test.min())
#y = digits.target
lb = LabelBinarizer()
y = lb.fit_transform(y)

#Train test split
#trainX, testX, trainY, testY = train_test_split(X, y, train_size=0.75)

#Defining neural network
#nn = NeuralNetwork(layers = [trainX.shape[1], 32, 16, 10], alpha=0.1)
nn = NeuralNetwork(layers = [X.shape[1],256,128,10], alpha=0.1)
print(nn)

#Training the model
#nn.fit(trainX, trainY)
nn.fit(X, y, epochs=40)

#Prediction
#predictions = nn.predict(testX)
predictions = nn.predict(data_test)
predictions = predictions.argmax(axis=1)
test_submission = np.c_[np.arange(1,len(predictions)+1), predictions]
np.savetxt("datasets/digit-recognizer/nn_mnist_test.txt", test_submission, fmt='%d', delimiter=",", header='ImageId,Label')
#print(classification_report(testY.argmax(axis=1), predictions))

