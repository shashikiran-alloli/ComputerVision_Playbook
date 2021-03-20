__author__ = "Shashikiran Alloli"
__version__ = "0.1"

'''Library Imports'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as k

'''Describing shallownet'''
class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        if k.image_data_format()=="channels_first":
            inputShape = (depth, height, width)

        model = Sequential()
        model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
