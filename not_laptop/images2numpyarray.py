'''
import os
import numpy as np
from keras.preprocessing import image

PATH = os.getcwd()

#train_path = PATH + '/data/train/'
train_path = '/Users/que/Data/data/train/'
train_batch = os.listdir(train_path)
x_train = []

# if data are in form of images
for sample in train_batch:
    if sample == '.DS_Store':
        continue
    img_path = train_path + sample
    x = image.load_img(img_path)
    # pre-processing if required
    x_train.append(x)

x_train = np.array(x_train)
print(x_train)



#test_path = PATH + '/data/test/'
test_path = '/Users/que/Data/data/test/'
test_batch = os.listdir(test_path)
x_test = []

for sample in test_batch:
    img_path = test_path + sample
    x = image.load_img(img_path)
    # pre-processing if required
    x_test.append(x)

# finally converting list into numpy array
x_test = np.array(x_test)

#print(x_test.length)

'''