import numpy as np
import cv2
from tcp_pickle_stream import listener
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

listen = listener('192.168.1.3')
model = ResNet50(weights='imagenet')
while(True):

    img = listen.get_frame()
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    predict_string = decode_predictions(preds, top=3)[0]
    predict_string = ''.join(str(predict_string[0]))

    cv2.putText(img, predict_string, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), lineType=cv2.LINE_AA)

    #if package:
        #cap = cv2.VideoCapture(0)
        # Capture frame-by-frame
        #ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        listen.close()
        break

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()

