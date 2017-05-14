#!/bin/bash
# This script is used to generate style video automatically.
# ICT: 2016-11-17
test $# -lt 4 && echo -e "The number of paramter is less than 2. Stop here.\
    \nUsage:\n\t./transfer_video.sh --model the_path_to_model --video video_name [--batch_size 8]" \
    && exit 0

##############################
# video to image
echo "***********************"
test ! -e $4 && echo "The video `$4` DO NOT exist" && exit 0

videoname=$(basename "$4")
videoname=${videoname%.*}

test ! -d video/${videoname} && mkdir -p video/${videoname}

ffmpeg -i $4 -q:v 1 video/${videoname}/${videoname}%06d.jpg


###############################
# Transfer frames
# Output path: output/feathers/ 
echo "***********************"
echo "Tranfer frames"

model=$2
content=video/${videoname}/
image_size=512
gpu=0

test ! -e $model && echo "The model `$model` DO NOT exist" && exit 0
if [ $# -ge 6 ];then
    batch_size=${6}
else
    batch_size=4    
fi

python gen_artwork.py --model ${model} --content ${content} --batch_size ${batch_size} --image_size ${image_size} --gpu ${gpu}



###############################
# Make Video 
echo "***********************"

dir=output/${model%.model}
frame=${videoname}

vb='10M' # video bitrate
r=30 # frame per sec

ffmpeg -f image2 -r ${r} -i ${dir}/${frame}%06d-styled.png -vcodec mpeg4 -vb ${vb} -y ./output/${frame}-styled.mp4
