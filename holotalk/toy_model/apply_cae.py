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

source_folder = '/home/whitejet/Datasets/experiments/faces/processed_subset'

# load and process the training/test data with load_img
start_load = time.time()
test_image = []
for i in os.listdir(source_folder+'/test'):
    img = image.load_img(source_folder+'/test/'+i,target_size=(72,80,1),grayscale=True)
    img = image.img_to_array(img) #type=float32
    img = img/255. #rescale to [0,1]
    #img = img.reshape(np.prod(img.shape[:])) #flatten
    test_image.append(img)
test_image = np.array(test_image)
#test_image = test_image.reshape((len(test_image), np.prod(test_image.shape[1:]))) #flatten
print('test data has shape = ',np.array(test_image).shape)
print('load_img takes time = ',time.time()-start_load)

# model I/O - model and weights all together
#loaded_model = load_model("AutoEncoder.h5")
loaded_model = load_model("AutoEncoder_C456DC456_100E.h5") #model no4, 100 epochs
loaded_model.summary()
# model validation

rand_int = random.randint(0,380)
decoded_imgs = loaded_model.predict(test_image[rand_int:rand_int+5])
print("Reconstructions take time = ",time.time()-start_load)

plt.figure(figsize=(72,80))
for i in range(5):
    #original
    ax = plt.subplot(2,5,i+1)
    plt.imshow(test_image[i+rand_int].reshape(72,80))  #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #reconstruction
    ax = plt.subplot(2,5,i+1+5)
    plt.imshow(decoded_imgs[i].reshape(72,80)) #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
