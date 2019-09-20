#!/usr/bin/bash
#use webcam to capture facial portrait and reconstruct with Convolutional AutoEncoder
cd holotalk
python webcam_capture.py
python webapply_cae.py
