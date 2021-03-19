__author__ = "Shashikiran Alloli"
__version__ = "0.1"

'''Library Imports'''
from skimage.exposure import rescale_intensity
from preprocessing import simplepreprocessor
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

'''Commandline arguments'''
argparser = argparse.ArgumentParser()
argparser.add_argument("-i","--image",required = True, help = "Path to image file")
args = vars(argparser.parse_args())

'''Convolution function'''
def convolution(image, K):
    iH, iW = image.shape[:2]
    kH, kW = K.shape[:2]

    pad = (kW - 1)//2

    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW),dtype="float")

    for y in np.arange(pad, iH):
        for x in np.arange(pad, iW):
            roi = image[y-pad:y+pad+1, x-pad:x+pad+1]
            k=(roi*K).sum()
            output[y-pad, x-pad] = k

    output = rescale_intensity(output,in_range=(0,255))
    output = (output*255).astype('uint8')
    return output

'''Creating kernels'''
# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
[0, 1, 0],
[1, -4, 1],
[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array((
[-2, -1, 0],
[-1, 1, 1],
[0, 1, 2]), dtype="int")

# construct the kernel bank, a list of kernels we're going to apply
# using both our custom `convolve` function and OpenCV's `filter2D`
# function
kernelBank = (
("small_blur", smallBlur),
("large_blur", largeBlur),
("sharpen", sharpen),
("laplacian", laplacian),
("sobel_x", sobelX),
("sobel_y", sobelY),
("emboss", emboss))

'''Loading input image'''
spp = simplepreprocessor(200,200)
image = cv2.imread(args['image'])
image = spp.fit_transform(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''Looping over kernels'''
for (kernelName, K) in kernelBank:
	# apply the kernel to the grayscale image using both our custom
	# `convolve` function and OpenCV's `filter2D` function
	print("[INFO] applying {} kernel".format(kernelName))
	convolveOutput = convolution(gray, K)
	opencvOutput = cv2.filter2D(gray, -1, K)
	# show the output images
	cv2.imshow("Original", gray)
	cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

