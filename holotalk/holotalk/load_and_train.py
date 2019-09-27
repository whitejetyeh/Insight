'''
Convolutional AutoEncoder
HoloTalk's second proto type reconstructing 2D protraits
'''
import LRsymmetrizer as sym
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.models import Model, model_from_json, load_model
import os, time
import numpy as np
import matplotlib.pyplot as plt

print(tf.version.VERSION)
print(tf.keras.__version__)

source_folder = os.getcwd()+'/tinygray'

# load and process the training/test data with load_img
start_load = time.time()

sub_source_folder = source_folder+'/train'
train_image = []
for i in os.listdir(sub_source_folder):
    for ang in (0,10,20,30,40):
        if i.endswith('_%d.png' % ang):
            img = sym.feed_processor(i,ang,sub_source_folder)
            # img.shape = (320,144) for vertically stacked imgs of (h,w) = (160,144)
            img = img/255. #rescale to [0,1]
            train_image.append(img)
train_image = np.array(train_image)
# train_image.shape = (N,320,144) for N imgs in sub_source_folder
train_image = np.expand_dims(train_image,axis=3)#keras format needs a dimension for the color channel


sub_source_folder = source_folder+'/test'
test_image = []
for i in os.listdir(sub_source_folder):
    for ang in (0,10,20,30,40):
        if i.endswith('_%d.png' % ang):
            img = sym.feed_processor(i,ang,sub_source_folder)
            # img.shape = (320,144) for vertically stacked imgs of (h,w) = (160,144)
            img = img/255. #rescale to [0,1]
            test_image.append(img)
test_image = np.array(test_image)
# test_image.shape = (N,320,144) for N imgs in sub_source_folder
test_image = np.expand_dims(test_image,axis=3)#keras format needs a dimension for the color channel

print('load_img takes time = ',time.time()-start_load)


#loaded_model = load_model("small_HoloEncoder_C567DC765.h5")
loaded_model = load_model("HoloEncoder_C567DDDC765.h5")
loaded_model.summary()
# model training
start_training = time.time()
loaded_model.fit(train_image, train_image,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(test_image, test_image))
print('Training takes time = ',time.time()-start_training)

# model validation
decoded_imgs = loaded_model.predict(test_image[:4])
decoded_imgs = loaded_model.predict(decoded_imgs)
decoded_imgs = loaded_model.predict(decoded_imgs)
decoded_imgs = loaded_model.predict(decoded_imgs)

plt.figure(figsize=(144,160))
for i in range(4):
    #original
    ax = plt.subplot(2,4,i+1)
    plt.imshow(test_image[i].reshape(160,72))  #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #reconstruction
    ax = plt.subplot(2,4,i+1+4)
    plt.imshow(decoded_imgs[i].reshape(160,72)) #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('validation_img.png')
