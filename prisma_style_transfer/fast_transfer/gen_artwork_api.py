#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
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

        # start_time = time.time()
        with tf.Session(graph=self.graph) as sess:
            content_images = tf.convert_to_tensor(input_image)
            images = tf.pack([content_images])

            input_node = sess.graph.get_tensor_by_name("input_node:0")
            output_node = sess.graph.get_tensor_by_name("output_node:0")

            im_images = sess.run(images)
            output = sess.run(output_node, feed_dict={input_node: im_images})

            output = np.squeeze(output, axis=0)
            # Just Used to test api
            # out_path = os.path.join(output_path, input_name + '-styled.png')
            # print('------------------------------------')
            # print "Save result in: ", out_path
            # print "Time for one image:", time.time() - start_time, "sec"
            # print('Finished!')

            # misc.imsave("./output", output[0])

            return output


def main(argv=None):
    """Set cuda visible device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    start_time = time.time()
    # load image
    im = Image.open(FLAGS.content_image).convert('RGB')
    im_n = np.asarray(im)
    print "Time: get image with Image", time.time() - start_time

    # create model
    start_time = time.time()
    transfer = TransferNet(FLAGS.model)
    print "Time: create class", time.time() - start_time

    # transfer image
    transfer.gen_single(im_n)

if __name__ == '__main__':
    tf.app.run()
