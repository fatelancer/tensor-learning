
import tensorflow as tf

import time

FLAGS = tf.app.flags.FLAGS

'''
Some util function

'''

def scalar_variable_summaries(var, name):
    """Attach summaries to a Scalar."""
    with tf.name_scope('summaries'):
        scalar_s = tf.summary.scalar(name, var)
        hist_s = tf.summary.histogram('hist-' + name, var)
    return [scalar_s, hist_s]


def get_model_suffix():
    model_suffix = "_"
    model_suffix += "arbi" if "arbi" in FLAGS.mode else "base"
    model_suffix += "_" + FLAGS.special_tag
    model_suffix += "_cw" + str(FLAGS.content_weight) + "_"
    model_suffix += "sw" + str(FLAGS.style_weight) + "_"
    model_suffix += "tw" + str(FLAGS.tv_weight) + "_"
    model_suffix += "ep" + str(FLAGS.epoch) + "_"
    model_suffix += "size" + str(FLAGS.image_size) + "_"
    model_suffix += "b" + str(int(FLAGS.batch_size))
    model_suffix += "_liuyi"
    return model_suffix

# Done
def log_train_configs(train_start, model_name, summ_path):
    with open(model_name + "_config.log", "a") as log_f:
        log_f.write("#-----------------------------------\n")
        log_f.write("Train start: " + time.ctime(train_start) + "\n")
        log_f.write("#-----------------------------------\n")

        log_f.write("Content weight: " + str(FLAGS.content_weight) + "\n")
        log_f.write("Style weight: " + str(FLAGS.style_weight) + "\n")
        log_f.write("Total variation weight: " + str(FLAGS.tv_weight) + "\n")
        log_f.write("Content layers: " + FLAGS.content_layers + "\n")
        log_f.write("Style layers: " + FLAGS.style_layers + "\n")
        log_f.write("Style scale: " + str(FLAGS.style_scale) + "\n")
        log_f.write("Style images: " + FLAGS.style_images + "\n")
        log_f.write("Batch size: " + str(FLAGS.batch_size) + "\n")
        log_f.write("Image size: " + str(FLAGS.image_size) + "\n")
        log_f.write("Train images path: " + FLAGS.train_images_path + "\n")
        log_f.write("Summary path: " + summ_path + "\n")
        log_f.write("VGG path: " + FLAGS.vgg_path + "\n")
        log_f.write("Learning rate: " + str(FLAGS.lr) + "\n")
        # log_f.write("GPU: " + str(FLAGS.gpu) + "\n")
        log_f.write("Epoch: " + str(FLAGS.epoch) + "\n")
        log_f.write("#-----------------------------------\n")