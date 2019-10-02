# take a portrait of you with your webcam
# to be used with flask
import cv2
import sys


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400)
#Webcams only support specific sets of widths & heights, e.g., 640x480

if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        print("pic taken successfully")
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('./model_folder/photo.jpg', frame)
    else:
        print("pic not taken correctly")
cv2.destroyAllWindows()
