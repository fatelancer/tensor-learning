#!/usr/bin/env python
"""Fast style transfer.
This file contains both training and generating code.

"""
from scipy import misc
import os
import time
import tensorflow as tf
import vgg
import model
import reader

# special tag
tf.app.flags.DEFINE_string("special_tag", "replicate_pad", "Special tag for model name")

# common arguments
tf.app.flags.DEFINE_string("mode", "train", "training-train or generate-gen")
tf.app.flags.DEFINE_integer("image_size", 256, "Size of output image")
tf.app.flags.DEFINE_string("gpu", "0", "Select which gpu to use, 0 or 1")

# arguments for train
tf.app.flags.DEFINE_float("content_weight", 9., "Weight for content features loss")
tf.app.flags.DEFINE_string("content_layers", "relu3_4", "Which VGG layer to extract content loss from")

tf.app.flags.DEFINE_float("tv_weight", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("vgg_path", "vgg19_36.mat", "Path to vgg model weights")

# tf.app.flags.DEFINE_string("model_path", None, "Path to write trained models")
tf.app.flags.DEFINE_string("model_name", None, "Name for output trained model")

tf.app.flags.DEFINE_string("train_images_path", "train2014", "Path to training images")

tf.app.flags.DEFINE_float("style_weight", 100., "Weight for style features loss")
tf.app.flags.DEFINE_string("style_layers", "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
                           "Which layers to extract style from")
tf.app.flags.DEFINE_string("style_images", "style.png", "Styles to train")
tf.app.flags.DEFINE_float("style_scale", 1, "Scale styles. Higher extracts smaller features")

tf.app.flags.DEFINE_integer("batch_size", 2, "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("summary_path", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_integer("epoch", 1e1, "Epochs for training")
tf.app.flags.DEFINE_float("lr", 1e-3, "learning rate for training")

# arguments for generate
tf.app.flags.DEFINE_string("model", "models/", "Path to read trained models")
tf.app.flags.DEFINE_string("content", None, "Path to content image(s)")
tf.app.flags.DEFINE_string("content_image", "toy.png", "Name for output styled image")
tf.app.flags.DEFINE_string("output", "style_output", "Name for output styled image")


FLAGS = tf.app.flags.FLAGS


def scalar_variable_summaries(var, name):
    """Attach summaries to a Scalar."""
    with tf.name_scope('summaries'):
        scalar_s = tf.scalar_summary(name, var)
        hist_s = tf.histogram_summary('hist-' + name, var)
    return [scalar_s, hist_s]


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) \
        - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1]))\
        - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])

    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))


def gram(layer):
    """ Get style with gram matrix. """
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size / FLAGS.batch_size)

    return grams


def get_style_features(style_paths, style_layers, net_type):
    with tf.Graph().as_default() as g:
        size = int(round(FLAGS.image_size * FLAGS.style_scale))
        images = tf.pack([reader.get_image(path, size) for path in style_paths])
        net, _ = vgg.net(FLAGS.vgg_path, images, net_type)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)


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
        log_f.write("GPU: " + str(FLAGS.gpu) + "\n")
        log_f.write("Epoch: " + str(FLAGS.epoch) + "\n")
        log_f.write("#-----------------------------------\n")


def perceptual_loss(net_type):
    """Compute perceptual loss of content and style"""
    # Set style image
    style_paths = FLAGS.style_images.split(',')
    # Set style layers and content layers in vgg net
    style_layers = FLAGS.style_layers.split(',')
    content_layers = FLAGS.content_layers.split(',')
    # Get style feature
    style_features_t = get_style_features(style_paths, style_layers, net_type)

    # Read images from dataset
    images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.train_images_path, epochs=FLAGS.epoch)
    # Transfer images
    generated = model.net(images / 255.)

    # Process generated and original images with vgg
    net, _ = vgg.net(FLAGS.vgg_path, tf.concat(0, [generated, images]), net_type)

    # Get content loss
    content_loss = 0
    for layer in content_layers:
        gen_features, images_features = tf.split(0, 2, net[layer])
        size = tf.size(gen_features)
        content_loss += tf.nn.l2_loss(gen_features - images_features) / tf.to_float(size)
    content_loss /= len(content_layers)

    # Get Style loss
    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        gen_features, _ = tf.split(0, 2, net[layer])
        size = tf.size(gen_features)
        # Calculate style loss for each style image
        for style_image in style_gram:
            style_loss += tf.nn.l2_loss(gram(gen_features) - style_image) / tf.to_float(size)
    style_loss /= len(style_layers)

    # Total loss
    total_v_loss = total_variation_loss(generated)
    loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * total_v_loss

    return generated, images, content_loss, style_loss, total_v_loss, loss


def gen_single():
    """ Transfer an image. """
    content_images = reader.get_image(FLAGS.content_image, FLAGS.image_size)
    images = tf.pack([content_images])
    generated_images = model.net(images / 255., if_train=False)

    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)

    # Ouput path
    model_p = FLAGS.model
    model_p = model_p if not model_p.endswith("/") else model_p[:-1]
    model_p = os.path.split(model_p)
    output_path = os.path.join("output", model_p[len(model_p)-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tf.Session() as sess:
        file_ = tf.train.latest_checkpoint(FLAGS.model)
        if not file_:
            print('Could not find trained model in {}'.format(FLAGS.model))
            return
        print('Using model from {}'.format(file_))

        # Get trained step
        index = file_.rfind("-")
        trained_step = file_[index:]

        saver = tf.train.Saver()
        saver.restore(sess, file_)
        
        print "Style image:", FLAGS.content_image
        start_time = time.time()
        
        # Run inference 
        images_t = sess.run(output_format)

        elapsed = time.time() - start_time
        print('Time: {}'.format(elapsed))

        out_path = os.path.join(output_path, FLAGS.output + trained_step + '.png')
        print "Save result in: ", out_path
        misc.imsave(out_path, images_t[0])
        
        print('------------------------------------')
        print('Finished!')

    return


def gen_from_directory():
    """ transfer images from a directory. """
    im_name = tf.placeholder(dtype=tf.string)
    im_format = tf.placeholder(dtype=tf.string)
    content_images = reader.get_image_frame(im_name, im_format, FLAGS.image_size)
    images = tf.pack([content_images])
    generated_images = model.net(images / 255., if_train=False)

    ims = os.listdir(FLAGS.content)
    im_nums = len(ims)
    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)

    # Ouput path
    model_p = FLAGS.model
    model_p = model_p if not model_p.endswith("/") else model_p[:-1]
    model_p = os.path.split(model_p)
    output_path = os.path.join("output", model_p[len(model_p)-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with tf.Session() as sess:
        file_ = tf.train.latest_checkpoint(FLAGS.model)
        if not file_:
            print('Could not find trained model in {}'.format(FLAGS.model))
            return
        print('Using model from {}'.format(file_))

        # Get trained step
        index = file_.rfind("-")
        trained_step = file_[index:]

        saver = tf.train.Saver()
        saver.restore(sess, file_)
        
        print "Transfer image:"
        start_time = time.time()
        
        # Run inference
        for i in range(im_nums):
            if ims[i].endswith(("png", "PNG")):
                format_ = "png"
            elif ims[i].endswith(("jpeg","jpg","JPEG","JPG",)):
                format_ = "jpg"
            else:
                print "---Unsupported image format:", ims[i]
                continue

            images_t = sess.run(output_format, feed_dict={im_name: FLAGS.content + ims[i],
                                                          im_format: format_})

            elapsed = time.time() - start_time
            start_time = time.time()
            print('Time: {}'.format(elapsed))
            format_index = ims[i].rfind('.')
            images_name = ims[i][:format_index]
            out_path = os.path.join(output_path, FLAGS.output + "-" 
                            + images_name + trained_step + '.png')
            print "Save result in: ", out_path
            misc.imsave(out_path, images_t[0], format="png")
        
        print('------------------------------------')
        print('Finished!')


def gen():
    """ transfer images from a directory. """
    content_images = reader.image(
            4,
            FLAGS.image_size,
            FLAGS.content,
            epochs=FLAGS.epoch,
            shuffle=False,
            crop=False)
    generated_images = model.net(content_images / 255.)

    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)
    with tf.Session() as sess:
        file_ = tf.train.latest_checkpoint(FLAGS.model)
        if not file_:
            print('Could not find trained model in {}'.format(FLAGS.model))
            return
        print('Using model from {}'.format(file_))
        saver = tf.train.Saver()
        saver.restore(sess, file_)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        start_time = time.time()
        try:
            while not coord.should_stop():
                print(i)
                images_t = sess.run(output_format)
                elapsed = time.time() - start_time
                start_time = time.time()
                print('Time for one batch: {}'.format(elapsed))

                for raw_image in images_t:
                    i += 1
                    print "Save result in: ", "output/"+FLAGS.output+'-{0:04d}.jpg'.format(i)
                    misc.imsave("output/" + FLAGS.output + '-{0:04d}.jpg'.format(i), raw_image)

        except tf.errors.OutOfRangeError:
            print('Done generate -- epoch limit reached!')

        except KeyboardInterrupt:
            print("Terminated by Keyboard Interrupt")

        finally:
            coord.request_stop()

        coord.join(threads)


def train(net_type):
    """ training a model for style transfer. """
    # Record training start time
    train_start = time.time()

    # Perceptual loss
    generated, images, content_loss, style_loss, total_v_loss, loss = perceptual_loss(net_type)

    global_step = tf.Variable(1, name="global_step", trainable=False)

    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, global_step=global_step)

    # # Decay learning rate according to global step
    # learning_rate = tf.train.exponential_decay(
    #     FLAGS.lr,  # Base learning rate.
    #     global_step,  # Current index into the dataset.
    #     1000,  # Decay step.s
    #     0.95,  # Decay rate.
    #     staircase=True)
    # # Use simple momentum for the optimization.
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    # Add summary
    content_loss_s = scalar_variable_summaries(content_loss, "content_loss")
    style_loss_s = scalar_variable_summaries(style_loss, "style_loss")
    tv_loss_s = scalar_variable_summaries(total_v_loss, "total_variation_loss")
    loss_s = scalar_variable_summaries(loss, "TOTAL_LOSS")
    # learning_rate_s = scalar_variable_summaries(learning_rate, "lr")

    merged = tf.merge_summary(content_loss_s + style_loss_s + tv_loss_s + loss_s)

    if FLAGS.batch_size <= 4:
        output_images = tf.saturate_cast(tf.concat(0, [generated, images]) + reader.mean_pixel, tf.uint8)
        # output_images = tf.saturate_cast(generated + reader.mean_pixel, tf.uint8)
    else:
        output_images = tf.saturate_cast(tf.concat(0, [tf.slice(generated, [0, 0, 0, 0], [4, -1, -1, -1]),
                                                       tf.slice(images, [0, 0, 0, 0], [4, -1, -1, -1])])
                                         + reader.mean_pixel, tf.uint8)

    # output_format = tf.saturate_cast(tf.concat(0, [generated, images]) + reader.mean_pixel, tf.uint8)
    # output_images = tf.saturate_cast(generated + reader.mean_pixel, tf.uint8)

    # Add output image summary
    im_summary = tf.image_summary("output-", output_images, max_images=8)
    im_merge = tf.merge_summary([im_summary])

    # Make output path
    model_suffix = "_" + FLAGS.special_tag
    model_suffix += "_cw" + str(FLAGS.content_weight) + "_"
    model_suffix += "sw" + str(FLAGS.style_weight) + "_"
    model_suffix += "tw" + str(FLAGS.tv_weight) + "_"
    model_suffix += "ss" + str(FLAGS.style_scale) + "_"
    model_suffix += "b" + str(int(FLAGS.batch_size))

    model_path = os.path.join("models", FLAGS.model_name + model_suffix)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = os.path.join(model_path, FLAGS.model_name)

    # Summary path
    summ_path = os.path.join(FLAGS.summary_path, FLAGS.model_name + model_suffix)
    if not os.path.exists(summ_path):
        os.makedirs(summ_path)

    # Record running configs in log file
    log_train_configs(train_start, model_name, summ_path)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        file_ = tf.train.latest_checkpoint(model_path)
        if file_:
            print('Restoring model from {}...'.format(file_))
            saver.restore(sess, file_)
        else:
            print('Initialize an new model...')
            sess.run(tf.initialize_all_variables())
        
        train_writer = tf.train.SummaryWriter(summ_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        total_time = 0
        step = 1
        try:
            while not coord.should_stop():
                if step % 20 == 0:
                    summary, _, loss_t, step = sess.run([merged, train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    start_time = time.time()
                
                    # Record summaries
                    train_writer.add_summary(summary, step)
                    print "# step, loss, elapsed time = ", step-1, loss_t, elapsed_time * 20

                else:
                    _, loss_t, step = sess.run([train_op, loss, global_step])
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    start_time = time.time()

                if step % 200 == 0:
                    im_summary = sess.run(im_merge)
                    train_writer.add_summary(im_summary, step)
                    # Save checkpoint file
                    saver.save(sess, model_name, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Finished training -- epoch limit reached!')

        except KeyboardInterrupt:
            print("Terminated by Keyboard Interrupt")

        finally:
            print('------------------------------------')
            print "Total time for", FLAGS.epoch, "epoch:", total_time
            coord.request_stop()

        coord.join(threads)
        
    # Log
    train_end = time.time()
    with open(model_name + "_config.log", "a") as log_f:
        log_f.write("Train end: " + time.ctime(train_end) + "\n")
        log_f.write("Total time: " + str(train_end - train_start) + " sec\n")
        log_f.write("#-----------------------------------\n\n")


def main(argv=None):
    """Set cuda visible device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    if FLAGS.mode == "gen":
        if FLAGS.content:
            gen_from_directory()
        elif FLAGS.content_image:
            gen_single()
        else:
            print "Please input content images path with arg: "
            print "\t\t--content content_images_path OR:"
            print "\t\t--content_image content_images"
    elif FLAGS.mode == "train":
        if FLAGS.vgg_path == "vgg19_36.mat":
            net_type = "vgg19"
        elif FLAGS.vgg_path == "vgg16_36.mat":
            net_type = "vgg16"
        else:
            print "Please specify valid vgg net model path."
            return
            
        train(net_type)


if __name__ == '__main__':
    tf.app.run()