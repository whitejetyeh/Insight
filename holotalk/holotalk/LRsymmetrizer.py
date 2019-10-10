'''
Symmertrizer replicates the right half part of human face to the left half part.
Wether the input yawing left or right, the feed_processor converts the input image
to be yawing left and combines it with the lefthand side image of people yawing left.
The output_processor converts the reconstructed input back to front, left/right side views.
'''
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def angle_conv(yaw):
    """
    round angles to integers by ten
    """
    yaw = np.rint(np.array(yaw)/10)*10
    return yaw.astype(int)

def feed_processor(image,yaw,source_folder,training=True):
    """
    Horizontally stack the input image with corresponding sideview from a model
    """
    #if training=False, user is reconstructing the side view from one model.
    img = cv2.imread(source_folder+'/'+image,0)# Using 0 to read image in grayscale mode
                             # cv2.imread output np.array(height,width,color channels)
    yaw = angle_conv(yaw)#convert angle to match data files
    str_yaw = '_'+str(yaw)
    if 0<=yaw<45:
        upper_img = img
        ang = '_'+str(yaw-90)
        if training:
            lower_img = cv2.imread(source_folder+'/'+image.replace(str_yaw,ang),0)# Used for training
        else:
            lower_img = cv2.imread(source_folder+'/110929151119'+ang+'.png',0)# Used for prediction
    elif -45<yaw<0:
        upper_img = cv2.flip(img, 1) #flip left to right
        ang = '_'+str(-yaw-90)
        if training:
            lower_img = cv2.imread(source_folder+'/'+image.replace(str_yaw,ang),0)# Used for training
        else:
            lower_img = cv2.imread(source_folder+'/110929151119'+ang+'.png',0)# Used for prediction
    else:
        print('turn too much, please retake photo.')
        return

    #check if image data is read succesfully
    if img is None:
        print('Photo is not read!')
        return
    elif lower_img is None:
        print('Missing training data!')

    #NEED to transverse img from (height,width,channels) to (width,height,channels) for keras
    return np.vstack((upper_img,lower_img))

def output_processor(image,yaw):
    """
    Split the stacked up image reconstruction back into frontal/left/right views
    """
    #image is the vertically stacked upper_img and lower_img.
    #input format = np.array(height,width)
    height = int(len(image)/2)
    upper_img = image[:height,:]
    lower_img = image[height:,:]
    yaw = angle_conv(yaw)#convert angle to match data files
    if yaw==0:
        front_screen = upper_img
        left_screen = lower_img
        right_screen = cv2.flip(lower_img, 1)
    elif 0<yaw<45:
        front_screen = upper_img
        left_screen = lower_img
        right_screen = 0*lower_img
    elif -45<yaw<0:
        front_screen = cv2.flip(upper_img, 1)
        left_screen = 0*lower_img
        right_screen = cv2.flip(lower_img, 1)
    return front_screen, left_screen, right_screen
