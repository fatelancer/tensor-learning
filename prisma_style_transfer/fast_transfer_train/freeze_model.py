#!/usr/bin/env python
"""Merge checkpoint and graph into a single protobuff file.
And gen artwork from pb model.
@author: jasvixban@gmail.com
"""
from scipy import misc
import os
import time
import tensorflow as tf
import vgg
import model
import reader
import freeze_graph

# arguments
tf.app.flags.DEFINE_string("mode","train","training-train or generate-gen")
tf.app.flags.DEFINE_integer("image_size", 512, "Size of output image")
tf.app.flags.DEFINE_string("gpu", "0", "Select which gpu to use, 0 or 1")

tf.app.flags.DEFINE_integer("batch_size", 2, "Number of concurrent images to train on")

tf.app.flags.DEFINE_string("model", "models/", "Path to read trained models")
tf.app.flags.DEFINE_string("content", None, "Path to content image(s)")
tf.app.flags.DEFINE_string("content_image", "toy.png", "Name for output styled image")
tf.app.flags.DEFINE_string("output", "style_output", "Name for output styled image")

tf.app.flags.DEFINE_string("in_graph_name", "in_graph_2.pb", "Name for output styled image")
tf.app.flags.DEFINE_string("out_graph_name", "out_graph_2.pb", "Name for output styled image")

FLAGS = tf.app.flags.FLAGS


def freeze_model():
    """ freeze graph. """
    input_node_names = "input_node"
    output_node_names = "output_node"
    
    content_images = reader.get_image(FLAGS.content_image, FLAGS.image_size)
    images = tf.pack([content_images])

    input_images = tf.placeholder(dtype=tf.float32, name=input_node_names)
    generated_images = model.net(input_images / 255., if_train=False)

    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8, name=output_node_names)
    
    with tf.Session() as sess:
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.model)
        if not checkpoint_file:
            print('Could not find trained model in {}'.format(FLAGS.model))
            return
        print('Using model from {}'.format(checkpoint_file))

        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        in_images = sess.run(images)
        images_t = sess.run(output_format, feed_dict={input_images: in_images})

        # Save graph
        tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model, FLAGS.in_graph_name)

    checkpoint_prefix = os.path.join(FLAGS.model, "saved_checkpoint")
    checkpoint_state_name = "checkpoint_state"

    # We save out the graph to disk, and then call the const conversion
    # routine.
    input_graph_path = os.path.join(FLAGS.model, FLAGS.in_graph_name)
    input_saver_def_path = ""
    input_binary = False
    
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(FLAGS.model, FLAGS.out_graph_name)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_file,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, "")

    print('------------------------------------')
    print('Finished!')


def unfreeze_model():
    """unfreeze model"""
    output_graph_path = os.path.join(FLAGS.model, FLAGS.out_graph_name)

    # Ouput path
    model_p = FLAGS.model
    model_p = model_p if not model_p.endswith("/") else model_p[:-1]
    model_p = os.path.split(model_p)
    output_path = os.path.join("output", model_p[len(model_p)-1])

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        content_images = reader.get_image(FLAGS.content_image, FLAGS.image_size)
        images = tf.pack([content_images])
        
        with tf.Session() as sess:
            input_node = sess.graph.get_tensor_by_name("input_node:0")
            output_node = sess.graph.get_tensor_by_name("output_node:0")

            im_images = sess.run(images)
            output = sess.run(output_node, feed_dict={input_node: im_images})

            out_path = os.path.join(output_path, FLAGS.output + '-unfreeze.png')
            print "Save result in: ", out_path
            print('------------------------------------')
            print('Finished!')
            
            misc.imsave(out_path, output[0])


def main(argv=None):
    """Set cuda visible device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    freeze_model()
    # unfreeze_model()

if __name__ == '__main__':
    tf.app.run()
