'''
 Facial pose estimation
 reference: https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV
 detect (Yaw, Roll, Pitch) angles from an image
--------------------------------------------------------------------------------
Modified "Facial Landmarks detection"
ref: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
use dlib.shape_predictor and dlib.get_frontal_face_dector to find landmarks
requirements: 1. pip install imutils; imutils: A series of convenience functions
to make basic image processing operations such as translation, rotation, resizing,
skeletonization, and displaying Matplotlib images easier with OpenCV and Python.
2. dlib’s pre-trained facial landmark detector
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
'''
#!!!!!!!!!! Need to add a flag if get_frontal_face_dector fails to detect!!!!!

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

def landmark_extraction(image, cropscale=1.8, shape_predictor="./holotalk/shape_predictor_68_face_landmarks.dat"):
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor)

	# input image size=640x480 by webcam_capture.py
	# load the input image and convert it to grayscale
	img = cv2.resize(cv2.imread(image),(640,480))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# crop a bigger bounding box
		dh = (cropscale-1)*(rect.bottom()-rect.top()) #scaled height difference
		dh = int(dh/2)
		dw = (cropscale-1)*(rect.right()-rect.left()) #scaled width difference
		dw = int(dw/2)
		crop_gray =	gray[rect.top()-dh:rect.bottom()+dh,rect.left()-dw:rect.right()+dw]
		cv2.imwrite('./static/img/face%d_crop.png' % i, cv2.resize(crop_gray,(72,80)))
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect) #68 (x,y) coordinates of landmarks
		shape = face_utils.shape_to_np(shape) #reshape to (68,2) np.array

		# define a dictionary that maps the indexes of the facial landmarks
		# These mappings are encoded inside the FACIAL_LANDMARKS_IDXS  dictionary
		# inside face_utils of the imutils library
	    # [[x,y],...]=[Nose tip, Chin, Left eye left corner, Right eye right corne, Left Mouth corner, Right mouth corner
		face_vector = np.array([shape[30],shape[8],shape[36],shape[45],shape[48],shape[54]],dtype=np.float32)
	return face_vector

# image_points = face_vector = landmark_extraction("image path")
# size = (height, width, color_channel)
def face_orientation(image_points, size):
    #universal facial 3d model points
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ],dtype=np.float32)

    # Camera internals

    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)


    axis = np.float32([[500,0,0],
                          [0,500,0],
                          [0,0,500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]


    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    #return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])
    return (int(roll), int(pitch), int(yaw))

#example of use
face_vector =  landmark_extraction("./static/img/photo.jpg")
print(face_vector)
print(face_orientation(face_vector, [480, 640]))
