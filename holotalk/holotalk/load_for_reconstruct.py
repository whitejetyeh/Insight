'''
Convolutional AutoEncoder
Loading CAE for HoloTalk to reconstruct 3D protraits from photo.jpg
'''
import LRsymmetrizer as sym
from pose_angles import landmark_extraction, face_orientation
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.models import Model, model_from_json, load_model
import os, time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()

def averaging(imgIn,imgOut,Rp=0.0):
    """
    averaging input and reconstruction
    """
    # Rp = [0,1], ratio of propagating how much input to reconstruction
    # Rp = 1/0 for maximully/minimumaly passing
    average = (Rp*imgIn+(1-Rp)*imgOut)/2
    return average

# facial pose estimation
face_vector =  landmark_extraction('./images/photo.jpg')
roll,pitch,yaw = face_orientation(face_vector, [480, 640])
yaw = -yaw#horizontal rotataion
print(roll,pitch,yaw)
source_img = 'face0_crop_dlib.png'
source_folder = os.getcwd()+'/images'

image = np.array([sym.feed_processor(source_img,yaw,source_folder,training=False)/255., ])
# image.shape = (1,128,64) for vertically stacked imgs of (h,w) = (64,64)
#rescale to [0,1]
image = np.expand_dims(image,axis=3)#keras format needs a dimension for the color channel

#loaded_model = load_model("small_HoloEncoder_C567DC765.h5")
loaded_model = load_model("argHoloEncoder_C56789DDC98765.h5")
loaded_model.summary()
print('Loading model takes time = ',time.time()-start_time)
start_predict = time.time()
# model validation; applied contrastive divergence learning by repeating predictions
decoded_imgs = loaded_model.predict(image)
decoded_imgs = loaded_model.predict(decoded_imgs)
decoded_imgs = loaded_model.predict(decoded_imgs)
# process output with left/right symmetry
front_screen, left_screen, right_screen = sym.output_processor(decoded_imgs.reshape(128,64), yaw)

# averaging input and reconstruction
pre_front_screen, pre_left_screen, pre_right_screen = sym.output_processor(image.reshape(128,64), yaw)
front_screen = averaging(pre_front_screen,front_screen, Rp=0)
left_screen = averaging(pre_left_screen,left_screen, Rp=0)
right_screen = averaging(pre_right_screen,right_screen, Rp=0)

print('Reconstruction takes time = ',time.time()-start_predict)

# output generated side views
plt.imsave('demoF.png',front_screen,cmap='Greys_r')
plt.imsave('demoL.png',left_screen,cmap='Greys_r')
plt.imsave('demoR.png',right_screen,cmap='Greys_r')

plt.figure(figsize=(18,20))
ax = plt.subplot(2,3,1)
plt.imshow(decoded_imgs[0,:,:,0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#original
ax = plt.subplot(2,3,2)
plt.imshow(plt.imread(source_folder+'/'+source_img))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(2,3,3)
plt.imshow(image[0,:,:,0])
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#reconstruction
ax = plt.subplot(2,3,4)
plt.imshow(left_screen.reshape(64,64)) #reshape from flatten&grayscale
plt.gray()
plt.title('left screen')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(2,3,5)
plt.imshow(front_screen.reshape(64,64)) #reshape from flatten&grayscale
plt.gray()
plt.title('front screen')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = plt.subplot(2,3,6)
plt.imshow(right_screen.reshape(64,64)) #reshape from flatten&grayscale
plt.gray()
plt.title('right screen')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig('demo_img.png')
