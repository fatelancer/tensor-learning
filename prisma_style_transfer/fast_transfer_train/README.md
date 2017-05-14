## Settings
- `git clone git@jas-Node5:~/srv/fast_transfer.git`
- Download COCOS dataset

## Requirement
- Tensorflow
- ffmpeg

## Usage

### Train
  - `python transfer_net.py --mode train --style_images [style_file_name]
    --train_images_path ~/DataSet/train2014/
    --model_name [style_name] --epoch 5 --batch_size 1 --gpu 0`
#### +Note:
  * [style_file_name] files in style/ directory e.g: ./style/candy.jpg

### Gen
  - transfer one image once
  `python transfer_net.py --mode gen --model [model_directory]
  --content_image [content_image_name]
  --output [output_filename] --image_size 1024 --gpu 0`
  - transfer images from directory
  `python transfer_net.py --mode gen
  --model models/[model_directory]
  --content [content_image_dir]
  --output [output_filename] --image_size 1024 --gpu 0`

#### +Note:
  - [content_image_name] example: input/girl.png
  - [content_image_dir] example: video/gosip-cut-rm/ 

### Freeze model
  - `python freeze_model.py --content_image input/png/live1.png
     --model models/dream28_replicate_pad_0_sw0_tw1e-05_ss3.0_b1/
     --out_graph_name dream.pb`

  - Output model is in models/dream28_replicate_pad_0_sw0_tw1e-05_ss3.0_b1/

## Train data dir:

### coco-train2014
  - jas-Node5:/home/jas/DataSet/train2014/ 
  - jas-Node4:/home/jas/cocos8.9/train2014
### cocos-valid2014   
  - jas-Node5:/home/jas/Installation/chainer-fast-neuralstyle/database/  
  - jas-Node4:/home/jas/Installation/chainer-fast-neuralstyle/database/

