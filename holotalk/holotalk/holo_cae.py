'''
Convolutional AutoEncoder for HoloTalk
HoloTalk's third proto type reconstructing 2D protraits
'''
import LRsymmetrizer as sym
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing import image
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
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

'''
# model building C567- Convolutional AutoEncoder_C567DDDC765
# compression rate = 1/4 in the dense layer
# input 2d images

#input placeholder
input_img = Input(shape=(160,72,1))
#convolutional structure
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#encoding structure
x = Flatten()(x)
encoded = Dense(5760, activation='relu')(x)
encoded = Dense(2880, activation='relu')(encoded)
encoded = Dense(5760, activation='relu')(encoded)
#decoding structure
x = Reshape((20,9,32))(encoded)
#deconvolutional structure
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
'''

# model building C56789- Convolutional AutoEncoder_C567DDDC765
# compression rate = 1/4 in the dense layer
# input 2d images

#input placeholder
input_img = Input(shape=(128,64,1))
#convolutional structure
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#encoding structure
x = Flatten()(x)
encoded = Dense(1024, activation='relu')(x)
encoded = Dense(1024, activation='relu')(encoded)
#decoding structure
x = Reshape((4,2,128))(encoded)
#deconvolutional structure
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# model training
start_training = time.time()
autoencoder.fit(train_image, train_image,
                epochs=10,
                batch_size=8,
                shuffle=True,
                validation_data=(test_image, test_image))
print('Training takes time = ',time.time()-start_training)

# model I/O - model and weights all together
autoencoder.save("HoloEncoder_C56789DDC98765.h5")
loaded_model = load_model("HoloEncoder_C56789DDC98765.h5")
loaded_model.summary()

# model validation
decoded_imgs = loaded_model.predict(test_image[:4])
decoded_imgs = loaded_model.predict(decoded_imgs)
decoded_imgs = loaded_model.predict(decoded_imgs)
decoded_imgs = loaded_model.predict(decoded_imgs)

plt.figure(figsize=(32,16))
for i in range(4):
    #original
    ax = plt.subplot(2,4,i+1)
    plt.imshow(test_image[i].reshape(128,64))  #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #reconstruction
    ax = plt.subplot(2,4,i+1+4)
    plt.imshow(decoded_imgs[i].reshape(128,64)) #reshape from flatten&grayscale
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('validation_img.png')
