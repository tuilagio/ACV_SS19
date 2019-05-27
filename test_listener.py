import numpy as np
import cv2
from tcp_pickle_stream import listener


listen = listener('localhost')
while(True):

    package = listen.get_frame()
    #if package:
        #cap = cv2.VideoCapture(0)
        # Capture frame-by-frame
        #ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
    cv2.imshow('frame', package)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        listen.close()
        break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()
