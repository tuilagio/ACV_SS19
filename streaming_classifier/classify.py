from keras.models import load_model, Sequential
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import numpy as np
import os
from keras.utils import plot_model


model = load_model('not-laptop-on-mac-75.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

test_laptops_dir = 'testing_data/laptops'
test_notlaptops_dir = 'testing_data/notlaptops'

correct_guesses, incorrect_guesses, counter = 0, 0, 0

for file in os.listdir(test_laptops_dir):
    img_path = test_laptops_dir + '/' + file
    test_image = image.load_img(img_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    # test_image = preprocess_input(test_image)

    # Predict the result
    result = model.predict(test_image)
    y_pred = model.predict_classes(test_image, 1, verbose=0)
    # print(y_pred)
    # print(result)

    # Print(result)
    if (result[0][0] == 0):
        print("Laptop")
        correct_guesses += 1
    if (result[0][0] == 1):
        print("Not laptop")
        incorrect_guesses  += 1
    counter += 1

print("Counter:", counter)
print("Correct guesses:", correct_guesses)
print("Incorrect guesses:", incorrect_guesses)
