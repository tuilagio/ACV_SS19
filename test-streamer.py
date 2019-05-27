import numpy as np
import cv2
from tcp_pickle_stream import streamer
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

stream = streamer('localhost')
cap = cv2.VideoCapture('big_buck_bunny.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        stream.send_frame(frame)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stream.close()
        break

# When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()