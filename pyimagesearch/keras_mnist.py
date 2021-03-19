__author__ = "Shashikiran Alloli"
__version__ = "0.1"

#Library imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as k
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import argparse

'''Commandline arguments'''
argparser = argparse.ArgumentParser()
argparser.add_argument("-o", "--output", required=True, help="Path to save trqa")
args = vars(argparser.parse_args())

'''Data load and manipulation'''
#loading mnist data
((trainX, trainY), (testX, testY)) = mnist.load_data()
#reshape
trainX = trainX.reshape((trainX.shape[0], 28*28*1))
testX = testX.reshape((testX.shape[0], 28*28*1))
#min-max normalize
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0
print("Shape of trainX ->",trainX.shape, "and shape of testX ->", testX.shape)
#one-hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
#kaggle mnist test data load
test_sub = np.genfromtxt("datasets/digit-recognizer/test.csv",dtype='float',delimiter=",",skip_header=1)
test_sub = test_sub/255.0


'''Neural Network Configuration'''
#Configuring layers of neural network
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

#Configuring optimizer
sgd = SGD(learning_rate=0.01)

#Compiling the model
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
print("[INFO]: Network information --")
print(model)

'''Training the model'''
H = model.fit(trainX, trainY, batch_size=128, epochs=100, validation_data=(testX, testY))

'''Predictions over test data'''
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

'''Saving the loss/accuracy over epochs plot on disk'''
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

'''Prediction for kaggle mnist competition'''
test_pred = model.predict(test_sub, batch_size=128)
test_pred = np.c_[np.arange(1,len(test_pred)+1),test_pred.argmax(axis=1)]
np.savetxt("datasets/digit-recognizer/test_submission.csv",test_pred,fmt="%d",delimiter=",",header='ImageId,Label')
