#!/usr/bin/env python
"""
Gen artwork from trained model.
Usage:
    1. Gen a single image
        python gen_artwork.py --model feathers.model --content_image girl.png
    2. Gen video frames
        python gen_artwork.py --model feathers.model --content video/
"""
from scipy import misc
import os
import time
import numpy as np
import tensorflow as tf
from . import reader
from PIL import Image

# arguments
tf.app.flags.DEFINE_integer("image_size", 512, "Size of output image")
tf.app.flags.DEFINE_string("gpu", "0", "Select which gpu to use, 0 or 1")
tf.app.flags.DEFINE_integer("batch_size", 2, "Number of concurrent images to transfer")

tf.app.flags.DEFINE_string("content", None, "Path to content image(s)")
tf.app.flags.DEFINE_string("content_image", "toy.png", "Name for input content image")

tf.app.flags.DEFINE_string("model", "style.model", "model file")

FLAGS = tf.app.flags.FLAGS


class TransferNet(object):
    def __init__(self, model_path):
        with tf.Graph().as_default() as graph:
            output_graph_def = tf.GraphDef()
            with open(model_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            self.graph = graph

    def gen_single(self, input_image):
        """Gen one styled image"""
        output_path = FLAGS.model
        output_path = (os.path.split(output_path))[-1]
        output_path = os.path.join("output", output_path[:output_path.rfind('.')])
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        input_name = (os.path.split(FLAGS.content_image))[-1]
        input_name = input_name[:input_name.rfind('.')]

        start_time = time.time()
        with tf.Session(graph=self.graph) as sess:
            content_images = tf.convert_to_tensor(input_image)
            images = tf.pack([content_images])

            input_node = sess.graph.get_tensor_by_name("input_node:0")
            output_node = sess.graph.get_tensor_by_name("output_node:0")

            im_images = sess.run(images)
            print("Time: get image", time.time() - start_time)

            start_time = time.time()
            output = sess.run(output_node, feed_dict={input_node: im_images})

            out_path = os.path.join(output_path, input_name + '-styled.png')
            print('------------------------------------')
            print("Save result in ", out_path)
            print("Time for one image:", time.time() - start_time, "sec")
            print('Finished!')

            misc.imsave(out_path, output[0])

            return im_images


def gen_from_directory():
    """Gen styled images from a directory"""
    img_dir = (FLAGS.content)[:-1] if (FLAGS.content).endswith('/') else FLAGS.content

    im_names = os.listdir(img_dir)
    ims = [x for x in im_names if os.path.isfile(img_dir + "/" + x)]
    im_nums = len(ims)
    batch_step = im_nums / FLAGS.batch_size
    rest_step = im_nums % FLAGS.batch_size
    if rest_step != 0:
        ims += ims[0:FLAGS.batch_size-rest_step]
        batch_step += 1
    ims_path = [ img_dir + "/" + x for x in ims ]

    step = 0

    # ouput styled images path
    output_path = FLAGS.model
    output_path = (os.path.split(output_path))[-1]
    output_path = os.path.join("output", output_path[:output_path.rfind('.')])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(FLAGS.model, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        # input data
        imgs_name = tf.placeholder(shape=(FLAGS.batch_size,), dtype=tf.string)
        img_format = "jpg" if ims[0].endswith(("jpg", "jpeg", "JPEG", "JPG")) else "png"
        content_images = reader.get_batch_images(imgs_name, img_format, FLAGS.image_size, FLAGS.batch_size)

        with tf.Session() as sess:
            
            input_node = sess.graph.get_tensor_by_name("input_node:0")
            output_node = sess.graph.get_tensor_by_name("output_node:0")

            print('------------------------------------')
            print("Total images:", im_nums)
            print("Total Batch:", batch_step)
            print("Start transfer>>>>>>")
            start_time = time.time()
            while step/FLAGS.batch_size < batch_step:
                content_ims = sess.run(content_images, feed_dict={imgs_name: ims_path[step:step + FLAGS.batch_size]})
                output = sess.run(output_node, feed_dict={input_node: content_ims})

                for j in range(FLAGS.batch_size):
                    out_path = os.path.join(output_path, ims[step+j][:-4] + '-styled.png')
                    misc.imsave(out_path, output[j])

                step += FLAGS.batch_size
                elapsed_time = time.time() - start_time
                start_time = time.time()

                print("Time for batch", step/FLAGS.batch_size, "with size =", FLAGS.batch_size, ":", elapsed_time, "sec")
        print('------------------------------------')
        print("Save result in: ", output_path)
        print('Finished!')


def main(argv=None):
    """Set cuda visible device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    # if FLAGS.content:
    #     gen_from_directory()
    # else:
    #     gen_single()

    # create model
    start_time = time.time()
    transfer = TransferNet(FLAGS.model)
    print("Time: create class", time.time() - start_time)

    start_time = time.time()
    # load image
    im = Image.open(FLAGS.content_image).convert('RGB')
    im_n = np.asarray(im)
    print("Time: get image with Image", time.time() - start_time)

    # transfer image
    transfer.gen_single(im_n)

if __name__ == '__main__':
    tf.app.run()
