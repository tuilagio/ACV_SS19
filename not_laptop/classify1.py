from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
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

img_path = 'dogsmiling.jpg'
img_path = '/Users/que/Data/ACV_SS19/catdog/training_data/dogs/dog000.jpg'
#img_path = '/Users/que/Data/ACV_SS19/catdog/training_data/cats/cat003.jpg'
import_path = '/Users/que/Downloads/vicious_dog_0.png'

from keras.models import load_model
model = load_model('model-dc.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing import image

test_image = image.load_img(img_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = preprocess_input(test_image)
#predict the result
result = model.predict(test_image)

#print(result)
if (result[0][0] == 1): print("dog")
if (result[0][0] == 0): print("cat")