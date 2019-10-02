'''
Convolutional AutoEncoder
Loading CAE for HoloTalk to reconstruct 3D protraits from face.png
'''
import holotalk.LRsymmetrizer as sym
from holotalk.pose_angles import landmark_extraction, face_orientation
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.models import Model, model_from_json, load_model
import os, time
import numpy as np
import matplotlib.pyplot as plt

def sideview_generation():
    # facial pose estimation
    face_vector =  landmark_extraction('./static/img/photo.jpg')
    roll,pitch,yaw = face_orientation(face_vector, [480, 640])
    yaw = -yaw#horizontal rotataion

    source_img = 'face0_crop.png'
    source_folder = './static/img'

    image = np.array([sym.feed_processor(source_img,yaw,source_folder,training=False)/255., ])
    # image.shape = (1,128,64) for vertically stacked imgs of (h,w) = (64,64)
    #rescale to [0,1]
    image = np.expand_dims(image,axis=3)#keras format needs a dimension for the color channel

    loaded_model = load_model("./holotalk/argHoloEncoder_C56789DDC98765.h5")
    loaded_model.summary()

    # model validation; applied contrastive divergence learning by repeating predictions
    decoded_imgs = loaded_model.predict(image)
    decoded_imgs = loaded_model.predict(decoded_imgs)
    decoded_imgs = loaded_model.predict(decoded_imgs)
    # process output with left/right symmetry
    front_screen, left_screen, right_screen = sym.output_processor(decoded_imgs.reshape(128,64), yaw)
    # output generated side views
    plt.imsave('./static/img/F.png',front_screen,cmap='Greys_r')
    plt.imsave('./static/img/L.png',left_screen,cmap='Greys_r')
    plt.imsave('./static/img/R.png',right_screen,cmap='Greys_r')

    return (roll,pitch,yaw)
