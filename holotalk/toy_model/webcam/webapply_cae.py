'''
reconstruct protraits with trained cae_proto
'''
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import random, time, os
import webcam_capture

webcam_capture.capture_face(cropscale=1.4)

source_folder = '/home/whitejet/Documents/webapp/holotalk/static/img'

# load and process the training/test data with load_img
start_load = time.time()
test_image = []
img = image.load_img(source_folder+'/face.jpg',target_size=(72,80,1),grayscale=True)
img = image.img_to_array(img) #type=float32
img = img/255. #rescale to [0,1]
#img = img.reshape(np.prod(img.shape[:])) #flatten
test_image.append(img)
test_image = np.array(test_image)
#test_image = test_image.reshape((len(test_image), np.prod(test_image.shape[1:]))) #flatten
print('test data has shape = ',np.array(test_image).shape)

# model I/O - model and weights all together
#loaded_model = load_model("AutoEncoder.h5")
loaded_model = load_model("AutoEncoder_C456DC456_100E.h5") #model no4, 100 epochs
loaded_model.summary()
# model reconstruction
start_predict = time.time()
decoded_imgs = loaded_model.predict(test_image)
decoded_imgs = loaded_model.predict(decoded_imgs)#multiple reconstruction with CD-k>1
decoded_imgs = loaded_model.predict(decoded_imgs)#multiple reconstruction with CD-k>1
decoded_imgs = loaded_model.predict(decoded_imgs)#multiple reconstruction with CD-k>1
decoded_imgs = loaded_model.predict(decoded_imgs)#multiple reconstruction with CD-k>1
print("Reconstructions take time = ",time.time()-start_predict)

plt.figure(figsize=(72,80))
#original
ax = plt.subplot(1,2,1)
plt.imshow(test_image[0].reshape(72,80))  #reshape from flatten&grayscale
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#reconstruction
ax = plt.subplot(1,2,2)
plt.imshow(decoded_imgs[0].reshape(72,80)) #reshape from flatten&grayscale
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.savefig(source_folder+'/reconstruction.jpg')
plt.show()
