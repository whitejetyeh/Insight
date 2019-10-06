# HOLOTALK

This project implements a convolutional autoencoder to generate 3D snapshots from one 2D frontal picture. The 3D portrait is holographically projectable by a [holographic pyramid](https://maker.pro/custom/projects/diy-hologram).

A webapp of [holotalk](http://matrixdata.club) has been established with flask. After uploading an image, a single user's face will be cropped out for a pose estimator to determine user's facial orientation, and the convolutional autoencoder will generate user's face viewing from left/right hand side.

## Introduction
Holotalk model consists of three main parts. The first part is a face detector based on the pretrained model, [dlib.get_frontal_face_dector](http://dlib.net/imaging.html#get_frontal_face_detector). The second part is a facial pose estimator. With another pretrained model, [dlib.shape_predictor](http://dlib.net/imaging.html#shape_predictor), to capture the coordinate of facial landmarks, one can estimate (roll, pitch, yaw) angles of the head pose based on an universal 3D facial model. The third part is an Convolutional AutoEncoder(CAE), which is trained to reconstruct the input of stacked up side views, and CAE is able to learn the correlations of a face viewing from different angles. Then, user's face is stacked with side views of a model with the same pose, and user's facial characteristics will propagate to the model through the correlations learned by CAE. At last, the modified side views of the model are returned as user's side views.

![HoloTalk demonstration](/images/HoloTalk_Demo.jpg)

## Demo Video
[![Webapp screen recording](/images/matrixdata_club.png)](https://youtu.be/IjoDcWxOqEs)
## Prerequistites
By exeucuting "pip install -r requirements.txt" with requirements.txt in the holotalk folder, the required libraries will be installed. The use of virtural environment is highly recommended.
## Image Processing
The training images are first processed by blender python api. In data/blender_snapshot.py, I rendered snapshots of the 3D facial model spinned at various angles, and then I resized the grayscale snapshots to (width,height)=(64,64). Then, in LRsymmetrizer.py, feed_processor stacks up the frontal image with the corresponding side image. By the assumption of symmetric face from left to right, I stacked the front view with the left view when the head is facing left, and vice versa. The pair of images is the training data for the convolutional autoencoder.
## How to run it
* The trained model is loaded for reconstrcting the side views of a detected face in photo.jpg in **load_for_reconstruct.py**.
* The Convolutional AutoEncoder is built in **holo_cae.py**; the encoder in this CAE has the following layers, 1. conv. with 32 channels + maxpooling 2. conv. with 64 channels + maxpooling 3. conv. with 128 channels + maxpooling 4. conv. with 256 channels + maxpooling 5. conv. with 512 channels + maxpooling 6. fully connected layer with 1024 nodes, and the decoder is simply the reverse of the encoder.
* In **load_and_train.py**, the model is loaded for more training epochs.

![HoloTalk demonstration](/images/model_pipeline.jpg)
## Author

## License

## Acknowledgmenets
### Dataset
### dlib
### pose estimator
