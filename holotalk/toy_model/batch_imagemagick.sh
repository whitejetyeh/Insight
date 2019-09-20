#!/usr/bin/bash
#batch process images with ImageMagick

source_folder="/home/whitejet/Datasets/experiments/faces/raw_subset/"
target_folder="/home/whitejet/Datasets/experiments/faces/processed_subsetD/"

#resize parameters
width=72
height=80

#resize source to target
cd $source_folder
for i in *.jpg;
do convert $i -resize $width!x$height! \
-colorspace Gray \
${target_folder}$i;
done
