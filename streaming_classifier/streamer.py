from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from tcp_pickle_stream import streamer
import numpy as np


stream = streamer('192.143.1.3') # The address of the host to where the images are sent
camera = PiCamera()
camera.resolution = (150, 150)
camera.framerate = 32 
rawCapture = PiRGBArray(camera, size=(150, 150))

# Allow the camera to warmup
time.sleep(0.1)

# Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    stream.send_frame(frame.array)
    key = cv2.waitKey(1) & 0xFF

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # If the `q` key is pressed, break
    if key == ord("q"):
        break

stream.close()
