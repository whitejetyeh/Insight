'''
Convolutional AutoEncoder
HoloTalk's proto type reconstructing 2D protraits
'''
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

source_folder = '/home/whitejet/Datasets/experiments/faces/processed_subsetD'

# load and process the training/test data with load_img
start_load = time.time()
train_image = []
for i in os.listdir(source_folder+'/train'):
    img = image.load_img(source_folder+'/train/'+i,target_size=(72,80,1),grayscale=True)
    img = image.img_to_array(img) #type=float32
    img = img/255. #rescale to [0,1]
    #img = img.reshape(np.prod(img.shape[:])) #flatten
    train_image.append(img)
train_image = np.array(train_image)
#train_image = train_image.reshape((len(train_image), np.prod(train_image.shape[1:]))) #flatten
test_image = []
for i in os.listdir(source_folder+'/test'):
    img = image.load_img(source_folder+'/test/'+i,target_size=(72,80,1),grayscale=True)
    img = image.img_to_array(img) #type=float32
    img = img/255. #rescale to [0,1]
    #img = img.reshape(np.prod(img.shape[:])) #flatten
    test_image.append(img)
test_image = np.array(test_image)
#test_image = test_image.reshape((len(test_image), np.prod(test_image.shape[1:]))) #flatten
print('training data has shape = ',np.array(train_image).shape)
print('load_img takes time = ',time.time()-start_load)
#print(train_image[0])
'''
# model building 1- AutoEncoder
# input flatten 1d array

#input placeholder
input_img = Input(shape=(5760,))
#encoder structure
encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
#decoder structure
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(5760, activation='sigmoid')(decoded) #reconstruction

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
'''

'''
# model building 2- Convolutional AutoEncoder_small
# input 2d images

#input placeholder
input_img = Input(shape=(72,80,1))
#encoder structure
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

#decoder structure
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
'''
'''
# model building 3- Convolutional AutoEncoder_flatten layer added
# input 2d images

#input placeholder
input_img = Input(shape=(72,80,1))
#convolutional structure
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#encoding structure
x = Flatten()(x)
x = Dense(2880, activation='relu')(x)
encoded = Dense(1440, activation='relu')(x)
#decoding structure
decoded = Dense(2880, activation='relu')(encoded)
x = Reshape((9,10,32))(decoded)
#deconvolutional structure
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
'''

# model building 4- Convolutional AutoEncoder_more channels
# input 2d images

#input placeholder
input_img = Input(shape=(72,80,1))
#convolutional structure
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#encoding structure
x = Flatten()(x)
encoded = Dense(2880, activation='relu')(x)
#decoding structure
x = Reshape((9,10,32))(encoded)
#deconvolutional structure
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# model training
start_training = time.time()
autoencoder.fit(train_image, train_image,
                epochs=100,
                batch_size=16,
                shuffle=True,
                validation_data=(test_image, test_image))
print('Training takes time = ',time.time()-start_training)

'''
# model I/O - seperated model and weights
start_saving = time.time()
saved_model = autoencoder.to_json()
with open("AutoEncoder.json","w") as json_file:
        json_file.write(saved_model)
autoencoder.save_weights("AutoEncoder.h5")
print("model saved as AutoEncoder")
json_file = open('AutoEncoder.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("AutoEncoder.h5")
loaded_model.compile(optimizer='adadelta', loss='binary_crossentropy') #need to compile again
print("model loaded from AutoEncoder")
print('Model S/L takes time = ',time.time()-start_saving)
'''

# model I/O - model and weights all together
autoencoder.save("AutoEncoder.h5")
loaded_model = load_model("AutoEncoder.h5")
loaded_model.summary()
# model validation

decoded_imgs = loaded_model.predict(test_image[:5])
plt.figure(figsize=(72,80))
for i in range(5):
    #original
    ax = plt.subplot(2,5,i+1)
    plt.imshow(test_image[i].reshape(72,80))  #reshape from flatten&grayscale
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
