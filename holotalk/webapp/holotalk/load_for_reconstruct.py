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
    row,pitch,yaw = face_orientation(face_vector, [480, 640])
    yaw = -yaw#horizontal rotataion

    source_img = './static/img/face0_crop.png'
    source_folder = './static/img'

    image = np.array([sym.feed_processor(source_img,yaw,source_folder)/255., ])
    # image.shape = (1,160,72) for vertically stacked imgs of (h,w) = (80,72)
    #rescale to [0,1]
    image = np.expand_dims(image,axis=3)#keras format needs a dimension for the color channel

    #loaded_model = load_model("small_HoloEncoder_C567DC765.h5")
    loaded_model = load_model("./holotalk/HoloEncoder_C567DDDC765.h5")
    loaded_model.summary()

    # model validation
    decoded_imgs = loaded_model.predict(image)
    #decoded_imgs = loaded_model.predict(decoded_imgs)
    #decoded_imgs = loaded_model.predict(decoded_imgs)
    #decoded_imgs = loaded_model.predict(decoded_imgs)

    front_screen, left_screen, right_screen = sym.output_processor(decoded_imgs.reshape(160,72), yaw)

    plt.figure(figsize=(18,20))
    #original
    ax = plt.subplot(2,3,2)
    plt.imshow(plt.imread(source_img))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #reconstruction
    ax = plt.subplot(2,3,4)
    plt.imshow(left_screen.reshape(80,72)) #reshape from flatten&grayscale
    plt.gray()
    plt.title('left screen')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,3,5)
    plt.imshow(front_screen.reshape(80,72)) #reshape from flatten&grayscale
    plt.gray()
    plt.title('front screen')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2,3,6)
    plt.imshow(right_screen.reshape(80,72)) #reshape from flatten&grayscale
    plt.gray()
    plt.title('right screen')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig('./static/img/demo_img.png')

    return (raw,pitch,yaw)
