__author__ = "Shashikiran Alloli"
__version__ = "0.1"

'''Library Imports'''
from preprocessing import simplepreprocessor
from preprocessing import imagetoarraypreprocessor
from datasets import simpledatasetloader
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

'''commandline arguments'''
argparser = argparse.ArgumentParser()
argparser.add_argument("-d", "--dataset", required=True, help="Path to input data")
argparser.add_argument("-o", "--output", required=True, help="Path to save loss/accuracy over epochs plot")
args = vars(argparser.parse_args())
images = list(paths.list_images(args['dataset']))

'''preprocessing'''
ita = imagetoarraypreprocessor()
spp = simplepreprocessor(32,32)
sdl = simpledatasetloader(preprocessors=[ita, spp])
data, labels = sdl.load(images, verbose=500)
data = data.astype('float')/255.0
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

'''train_test_split'''
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

'''building model'''
model = ShallowNet.build(32,32,3,len(lb.classes_))
sgd = SGD(learning_rate=0.01)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

'''training the model'''
H = model.fit(trainX, trainY, batch_size=32, epochs=100, validation_data=(testX, testY))

'''predictions'''
preds = model.predict(testX, batch_size=32)
preds = preds.argmax(axis=1)
print("Classes are ->",list(lb.classes_))
print(classification_report(testY.argmax(axis=1),preds,target_names=lb.classes_))

'''Plotting loss/accuracy over increasing epochs'''
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1,101),H.history['loss'],label='training_loss')
plt.plot(np.arange(1,101),H.history['val_loss'],label='validation_loss')
plt.plot(np.arange(1,101),H.history['accuracy'],label='training_accuracy')
plt.plot(np.arange(1,101),H.history['val_accuracy'],label='validation_accuracy')
plt.title('Loss/Accuracy over epochs')
plt.xlabel('# Epoch')
plt.ylabel('Loss/Accuracy')
plt.savefig(args['output'])

