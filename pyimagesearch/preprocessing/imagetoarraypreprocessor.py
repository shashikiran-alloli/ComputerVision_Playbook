__author__ = "Shashikiran Alloli"
__version__ = "0.1"

'''Library Imports'''
from tensorflow.keras.preprocessing.image import img_to_array

class imagetoarraypreprocessor():
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def fit_transform(self, image):
        return img_to_array(image, data_format=self.dataFormat)
