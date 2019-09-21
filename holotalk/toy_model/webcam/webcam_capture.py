# take a portrait of you with your webcam
# to be used with flask
import cv2
import sys

def capture_face(cropscale=1):
    cascPath = '/home/whitejet/anaconda3/envs/insight/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400)
    #Webcams only support specific sets of widths & heights, e.g., 640x480

    if cap.isOpened():
            ret, frame = cap.read()
            face = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in face:
                #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                dh = (cropscale-1)*h #scaled height difference
                dh = int(dh/2)
                dw = (cropscale-1)*w #scaled width difference
                dw = int(dw/2)
                crop_face = frame[y-dh:y+h+dh, x-dw:x+w+dw]
                crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2GRAY)
                crop_face = cv2.resize(crop_face,(72,80))
                cv2.imwrite('../static/img/face.jpg', crop_face)
                cap.release()
            if ret and frame is not None:
                print("pic taken successfully")
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('../static/img/photo.jpg', frame)
            else:
                print("pic not taken correctly")
            cv2.destroyAllWindows()
            return
#capture_face(cropscale=1.4)
