__author__="Shashikiran"
__version__="0.1"

'''
Library imports
'''
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from datasets.SimpleDatasetLoader import *
from preprocessing.SimplePreprocessor import *
from imutils import paths
import argparse

'''
Featching commandline arguments
'''
argparser = argparse.ArgumentParser()
argparser.add_argument("-d","--dataset",required=True,help="path to dataset")
argparser.add_argument("-a","--alpha",default=0.1,help="learning rate")
args = vars(argparser.parse_args())

'''
Parsing image paths
'''
images=list(paths.list_images(args['dataset']))

'''
Preprocessing
'''
spp = simplepreprocessor(width=32, height=32)
sdl = simpledatasetloader(preprocessors=[spp])
data, labels = sdl.load(images, verbose=5)
data = data.reshape(data.shape[0],3072)

'''
Label encoder
'''
le = LabelEncoder()
labels = le.fit_transform(labels)

'''
Train test split
'''
trainX, testX, trainY, testY = train_test_split(data, labels, train_size=0.75, random_state=42)

'''
Training SGDClassifier with l1 and l2
'''
for l in [None, "l1", "l2", "ElasticNet"]:
    print("Going with regularization parameter - {}".format(l))

    sgd = SGDClassifier(loss="log", penalty=l, max_iter=1000, tol=1e-3, n_jobs=-1, learning_rate='constant', eta0=0.01, random_state=25)
    sgd.fit(trainX, trainY)

    acc = sgd.score(testX, testY)
    print("{} penalty accuracy -> {}".format(l,acc))









