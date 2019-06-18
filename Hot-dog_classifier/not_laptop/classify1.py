from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_path = 'trying_data/63.tucano-shine-slim-tasche-fuer-macbook-pro-13-blau_z1_600x600.jpg'
#img_path = 'trying_data/63.Acer-Nitro-5-AN515-43-wp-03.jpg'
img_path = 'training_data/laptops/laptop000.jpg'
#img_path = 'training_data/notlaptops/notlaptop001.jpg'
img_path = 'trying_data/dogsmiling.jpg'


from keras.models import load_model
model = load_model('not-laptop-on-mac.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing import image

test_image = image.load_img(img_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = preprocess_input(test_image)

#predict the result
result = model.predict(test_image)
y_pred = model.predict_classes(test_image, 1, verbose=0)
print(y_pred)
print(result)
#print(result)
if (result[0][0] == 0): print("Laptop")
if (result[0][0] == 1): print("Not laptop")
