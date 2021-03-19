__author__ = "Shashikiran Alloli"
__version__ = "0.1"

'''Library Imports'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse

'''Commandline arguments'''
argparser = argparse.ArgumentParser()
argparser.add_argument("-o","--output",required=True, help="Output loss/accuracy plot to output image file")
args = vars(argparser.parse_args())

'''Load dataset'''
#Loading dataset
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#Reshaping
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

#min-max normalize
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

#One-Hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]

'''Network layers architecture'''
#Defining neural network
model = Sequential()
model.add(Dense(1024,input_shape=(3072,),activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))

#Defining optimizer
sgd = SGD(learning_rate=0.01)

#Compiling the network
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])

'''Training the network'''
H = model.fit(trainX, trainY, batch_size=32, epochs=100, validation_data=(testX, testY))

'''Predictions'''
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

'''Saving plot of loss/accuracy over epochs'''
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="training loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="validation loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="training accuracy")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="validation accuracy")
plt.title("Loss/Accuracy over epochs")
plt.xlabel("epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])


