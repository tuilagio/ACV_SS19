# Lets declare our image dimensions
# we are using coloured images.
nrows = 224
ncolumns = 224
channels = 3  # change to 1 if you want to use grayscale image
import cv2
import os
import numpy as np
import sys

def read_and_process_image(list_of_images, path):
    """
    Returns two arrays:
        X is an array of resized images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels

    for image in list_of_images:
        if image == '.DS_Store':
            continue
        print(image)
        X.append(cv2.resize(cv2.imread(path + image, cv2.IMREAD_COLOR), (nrows, ncolumns),
                            interpolation=cv2.INTER_CUBIC))  # Read the image
        # get the labels
        # if 'dog' in image:
        #     y.append(1)
        # elif 'cat' in image:
        #     y.append(0)
        y.append(image)

    return X, y


train_path = '/Users/que/Data/data/train/'
train_batch = os.listdir(train_path)
x_train = []
y_train = []
x_train, y_train = read_and_process_image(train_batch, train_path)

test_path = '/Users/que/Data/data/test/'
test_batch = os.listdir(test_path)
x_test = []
y_test = []
x_test, y_test = read_and_process_image(test_batch, test_path)



