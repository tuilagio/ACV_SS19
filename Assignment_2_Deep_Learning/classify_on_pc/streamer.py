from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from tcp_pickle_stream import streamer
#from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


stream = streamer('192.168.1.3')
#model = ResNet50(weights='imagenet')
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (224, 224)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(224, 224))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #img = frame.array
    #x = image.img_to_array(frame.array)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    #preds = model.predict(x)
    #predict_string = decode_predictions(preds, top=3)[0]
    #predict_string = ''.join(str(predict_string[0]))

    #cv2.putText(img, predict_string, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)
    stream.send_frame(frame.array)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

stream.close()
