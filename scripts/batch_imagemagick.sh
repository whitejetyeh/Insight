#!/usr/bin/bash
#batch process images with ImageMagick

source_folder="/home/whitejet/Datasets/experiments/holo_cae/two_persons/test/"
target_folder="/home/whitejet/Datasets/experiments/holo_cae/two_persons/minigray/test/"

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
