#!/usr/bin/bash
#batch process images with ImageMagick

source_folder="/home/whitejet/Datasets/experiments/3dto2d/"
target_folder="/home/whitejet/Datasets/experiments/3dto2d/processed/"

#resize parameters
width=144
height=160

#resize source to target
cd $source_folder
for i in *.png;
do convert $i -resize $width!x$height! \
-colorspace Gray \
${target_folder}$i;
done
