import numpy as np
import cv2
from tcp_pickle_stream import listener
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = load_model('not-laptop-on-mac.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

listen = listener('192.143.1.3') # This host's address

while(True):
    test_frame = listen.get_frame()
    #image_path = 'testing_data/notlaptops/notlaptop000.jpg'
    #test_image = image.load_img(image_path, target_size=(150, 150))
    #test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_frame, axis=0)
    test_image = preprocess_input(test_image)

    result = model.predict(test_image)
    y_pred = model.predict_classes(test_image, 1, verbose=0)
    #predict_string = decode_predictions(result, top=3)[0]
    #predict_string = ''.join(str(predict_string[0]))

    if (result[0][0] == 0): predict_string = "Laptop"
    if (result[0][0] == 1): predict_string = "Not laptop"

    # Display the resulting frame
    cv2.putText(test_frame, predict_string, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow('frame', test_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        listen.close()
        break

cv2.destroyAllWindows()
