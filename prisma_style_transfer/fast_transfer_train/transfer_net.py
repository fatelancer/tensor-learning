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
import utils

# special tag
tf.app.flags.DEFINE_string("special_tag", "replicate_pad", "Special tag for model name")

# common arguments
tf.app.flags.DEFINE_string("mode", "train", "training-train or generate-gen")
tf.app.flags.DEFINE_integer("image_size", 256, "Size of output image")
# tf.app.flags.DEFINE_string("gpu", "0", "Select which gpu to use, 0 or 1")

#######################################训练参数###########################################################
# 这里竟然使用relu3的结果, 而且权值的调整和原来的project相差比较大
tf.app.flags.DEFINE_float("content_weight", 1., "Weight for content features loss")
# 这里原来是用了 relu3-4
tf.app.flags.DEFINE_string("content_layers", "relu3_4", "Which VGG layer to extract content loss from")

# 只保留了需要用的最小部分
tf.app.flags.DEFINE_float("tv_weight", 100, "Weight for total variation loss")
tf.app.flags.DEFINE_string("vgg_path", "vgg19_36.mat", "Path to vgg model weights")

# tf.app.flags.DEFINE_string("model_path", None, "Path to write trained models")
tf.app.flags.DEFINE_string("model_name", None, "Name for output trained model")

tf.app.flags.DEFINE_string("train_images_path", "F:\\test_resize", "Path to training images")

tf.app.flags.DEFINE_float("style_weight", 1000., "Weight for style features loss")
# 这里倒是遵循了原来的选择，大概不那么好选择了
# 权值是平均分的
tf.app.flags.DEFINE_string("style_layers", "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1",
                           "Which layers to extract style from")

# 这里的多风格融合设计有点多余
tf.app.flags.DEFINE_string("style_images", "style/starry_night.png", "Styles to train")
tf.app.flags.DEFINE_float("style_scale", 1, "Scale styles. Higher extracts smaller features")

# 因为不太使用 batch Normalization 所以没什么所谓，不过大的batch应该可以加速并行计算
tf.app.flags.DEFINE_integer("batch_size", 1, "Number of concurrent images to train on")
tf.app.flags.DEFINE_string("summary_path", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_integer("epoch", 5, "Epochs for training")

# 在Batch Normalization的情况下可以调大一点, 但是对于Instance Normalization呢?
# 在 Loop 11000 我 调整为了 1e-1 之前是 1e-3
tf.app.flags.DEFINE_float("lr", 1e-3, "learning rate for training")

# checkpoint相关参数
# tf.app.flags.DEFINE_string("checkpoint_path", "checkpoint/%s.model" % get_time(), "use time to identify checkpoint")

tf.app.flags.DEFINE_integer("record_interval", 200, "the frequency to summary and refresh model recording")
tf.app.flags.DEFINE_integer("save_model_interval", 2000, "the frequency to summary and refresh model recording")
#####################################生成参数######################################################################
tf.app.flags.DEFINE_string("model", "models/", "Path to read trained models")
tf.app.flags.DEFINE_string("content", None, "Path to content image(s)")
tf.app.flags.DEFINE_string("content_image", "toy.png", "Name for output styled image")
tf.app.flags.DEFINE_string("output", "style_output", "Name for output styled image")


FLAGS = tf.app.flags.FLAGS



def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], [-1, height - 1, -1, -1]) \
        - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], [-1, -1, width - 1, -1])\
        - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])

    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))



# Done
def get_style_features(style_paths, style_layers, net_type):
    with tf.Graph().as_default() as g:
        size = int(round(FLAGS.image_size * FLAGS.style_scale))
        images = tf.stack([reader.get_image(path, size) for path in style_paths])
        net, _ = vgg.net(FLAGS.vgg_path, images, net_type)
        features = []
        for layer in style_layers:
            features.append(model.gram(net[layer], FLAGS.batch_size))

        with tf.Session() as sess:
            return sess.run(features)


def perceptual_loss(net_type):
    """Compute perceptual loss of content and style

    Return:
        generated 前向生成网络
        images 输入图片(batch based)
        loss 各种loss.
    """
    # Set style image
    style_paths = FLAGS.style_images.split(',')
    # Set style layers and content layers in vgg net
    style_layers = FLAGS.style_layers.split(',')
    content_layers = FLAGS.content_layers.split(',')
    # Get style feature, pre calculated and save it in memory
    style_features_t = get_style_features(style_paths, style_layers, net_type)

    # Read images from dataset
    images = reader.image(FLAGS.batch_size, FLAGS.image_size, FLAGS.train_images_path, epochs=FLAGS.epoch)

    # Transfer images
    # 为什么要换成0-1编码?
    # 这里和里面的处理对应起来, 虽然这么写很丑， 也容易忘
    generated = model.net(images / 255)
    # generated = model.net(tf.truncated_normal(images.get_shape(), stddev=0.3))


    # Process generated and original images with vgg
    net, _ = vgg.net(FLAGS.vgg_path, tf.concat([generated, images], 0), net_type)

    # Get content loss
    content_loss = 0
    for layer in content_layers:
        # 平均分为两组，每组都是batch长度的图片组
        gen_features, images_features = tf.split(net[layer], num_or_size_splits=2, axis=0)
        size = tf.size(gen_features)
        content_loss += tf.nn.l2_loss(gen_features - images_features) / tf.to_float(size)
    content_loss /= len(content_layers)

    # Get Style loss
    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        gen_features, _ = tf.split(net[layer], num_or_size_splits=2, axis=0)
        size = tf.size(gen_features)
        # Calculate style loss for each style image
        for style_image in style_gram:
            style_loss += tf.nn.l2_loss(model.gram(gen_features, FLAGS.batch_size) - style_image) / tf.to_float(size)
    style_loss /= len(style_layers)

    # Total loss
    total_v_loss = total_variation_loss(generated)
    loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * total_v_loss

    return generated, images, content_loss, style_loss, total_v_loss, loss


def gen_single():
    """ Transfer an image. """

    content_images = reader.get_image(FLAGS.content_image, FLAGS.image_size)
    images = tf.stack([content_images])
    generated_images = model.net(images / 255., if_train=False)

    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)

    # Output path

    model_path = os.path.join('models', FLAGS.model_name + utils.get_model_suffix())
    ### model_p = model_p if not model_p.endswith("/") else model_p[:-1]
    ### model_p = os.path.split(model_p)
    output_path = os.path.join("output", FLAGS.model_name + utils.get_model_suffix())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tf.Session() as sess:
        file_ = tf.train.latest_checkpoint(model_path)
        if not file_:
            print('Could not find trained model in {}'.format(model_path))
            return
        print('Using model from {}'.format(file_))

        # Get trained step
        index = file_.rfind("-")
        trained_step = file_[index:]

        saver = tf.train.Saver()
        saver.restore(sess, file_)
        
        print("Style image:", FLAGS.content_image)
        start_time = time.time()
        
        # Run inference 
        images_t = sess.run(output_format)

        elapsed = time.time() - start_time
        print('Time: {}'.format(elapsed))

        out_path = os.path.join(output_path, FLAGS.output + trained_step + '-' + str(int(time.time())) + '.jpg')
        print("Save result in: ", out_path)
        misc.imsave(out_path, images_t[0])
        
        print('------------------------------------')
        print('Finished!')

    return


def gen_from_directory():
    """ transfer images from a directory. """
    im_name = tf.placeholder(dtype=tf.string)
    im_format = tf.placeholder(dtype=tf.string)
    content_images = reader.get_image_frame(im_name, im_format, FLAGS.image_size)
    images = tf.stack([content_images])
    generated_images = model.net(images / 255., if_train=False)

    ims = os.listdir(FLAGS.content)
    im_nums = len(ims)
    output_format = tf.saturate_cast(generated_images + reader.mean_pixel, tf.uint8)

    # Ouput path
    model_path = os.path.join('models', FLAGS.model_name + utils.get_model_suffix())
    ### model_p = model_p if not model_p.endswith("/") else model_p[:-1]
    ### model_p = os.path.split(model_p)
    output_path = os.path.join("output", FLAGS.model_name + utils.get_model_suffix())

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with tf.Session() as sess:
        file_ = tf.train.latest_checkpoint(model_path)
        if not file_:
            print('Could not find trained model in {}'.format(FLAGS.model))
            return
        print('Using model from {}'.format(file_))

        # Get trained step
        index = file_.rfind("-")
        trained_step = file_[index:]

        saver = tf.train.Saver()
        saver.restore(sess, file_)
        
        print("Transfer image:")
        start_time = time.time()
        
        # Run inference
        for i in range(im_nums):
            if ims[i].endswith(("png", "PNG")):
                format_ = "png"
            elif ims[i].endswith(("jpeg","jpg","JPEG","JPG",)):
                format_ = "jpg"
            else:
                print("---Unsupported image format:", ims[i])
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
            print("Save result in: ", out_path)
            misc.imsave(out_path, images_t[0], format="png")
        
        print('------------------------------------')
        print('Finished!')

# duplicated with above function
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
                    print("Save result in: ", "output/"+FLAGS.output+'-{0:04d}.jpg'.format(i))
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

    global_step = tf.Variable(1, name="global_step", trainable=False)

    # Perceptual loss
    generated, images, content_loss, style_loss, total_v_loss, loss = perceptual_loss(net_type)

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
    content_loss_summary = utils.scalar_variable_summaries(content_loss, "content_loss")
    style_loss_summary = utils.scalar_variable_summaries(style_loss, "style_loss")
    tv_loss_summary = utils.scalar_variable_summaries(total_v_loss, "total_variation_loss")
    loss_summary = utils.scalar_variable_summaries(loss, "TOTAL_LOSS")
    # learning_rate_s = scalar_variable_summaries(learning_rate, "lr")



    ### if FLAGS.batch_size <= 4:
    ###    output_images = tf.saturate_cast(tf.concat(0, [generated, images]) + reader.mean_pixel, tf.uint8)
        # output_images = tf.saturate_cast(generated + reader.mean_pixel, tf.uint8)
    ### else:
    ###    output_images = tf.saturate_cast(tf.concat(0, [tf.slice(generated, [0, 0, 0, 0], [4, -1, -1, -1]),
    ###                                                   tf.slice(images, [0, 0, 0, 0], [4, -1, -1, -1])])
    ###                                     + reader.mean_pixel, tf.uint8)

    output_format = tf.saturate_cast(tf.concat([generated, images], 0) + reader.mean_pixel, tf.uint8)
    # output_images = tf.saturate_cast(generated + reader.mean_pixel, tf.uint8)

    # Add output image summary
    image_summary = tf.summary.image("output-image", output_format, max_outputs=8)

    merge_summary = tf.summary.merge(content_loss_summary + style_loss_summary + tv_loss_summary + loss_summary + [image_summary])
    ### im_merge = tf.merge_summary([im_summary])

    # Make output path

    model_suffix = utils.get_model_suffix()
    model_path = os.path.join("models", FLAGS.model_name + model_suffix)
    best_model_path = os.path.join(model_path, "best_model")


    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = os.path.join(model_path, FLAGS.model_name)

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    best_model_name = os.path.join(best_model_path, FLAGS.model_name)

    # Summary path
    summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name + model_suffix)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # Record running configs in log file
    utils.log_train_configs(train_start, model_name, summary_path)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        best_saver = tf.train.Saver(tf.all_variables())

        file_ = tf.train.latest_checkpoint(model_path)
        if file_:
            print('Restoring model from {}...'.format(file_))
            saver.restore(sess, file_)
        else:
            print('Initialize an new model...')
            sess.run(tf.initialize_all_variables())

        sess.run(tf.initialize_local_variables())
        
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        elapsed_time = 0
        total_time = 0
        step = 1
        best_loss = float('inf')

        # print(sess.run(generated).shape)
        # print(sess.run(images).shape)
        # exit(1)

        while not coord.should_stop():
            try:
                _, c_loss, s_loss, tv_loss, total_loss, step = sess.run([train_op, content_loss, style_loss, total_v_loss, loss, global_step])

                if step % FLAGS.record_interval == 0:
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    start_time = time.time()
                    summary = sess.run(merge_summary)
                    # Record summaries
                    summary_writer.add_summary(summary, step)

                    print("===============Step %d ================" % step)
                    print("content_loss is %f" % c_loss)
                    print("style_loss is %f" % s_loss)
                    print("tv_loss is %f" % tv_loss)
                    print("total_loss is %f" % total_loss)
                    print("now, best_loss is %f" % best_loss)
                    print("Speed is %f s/loop" % (elapsed_time / FLAGS.record_interval))
                    print("===============================================")

                if step % FLAGS.save_model_interval == 0:
                    saver.save(sess, model_name, global_step=step)

                    if total_loss < best_loss:
                        # im_summary = sess.run(im_merge)
                        # train_writer.add_summary(im_summary, step)
                        # Save checkpoint file
                        best_saver.save(sess, best_model_name, global_step=step)
                        best_loss = total_loss




            except tf.errors.OutOfRangeError:
                print('Finished training -- epoch limit reached!')
                break

            except KeyboardInterrupt:
                print("Terminated by Keyboard Interrupt")
                break

            except:
                # some unknown error like disk reading error, just ignore it.
                continue


        print('------------------------------------')
        print("Total time for", FLAGS.epoch, "epoch:", total_time)
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if FLAGS.mode == "gen":
        if FLAGS.content:
            gen_from_directory()
        elif FLAGS.content_image:
            gen_single()
        else:
            print("Please input content images path with arg: ")
            print("\t\t--content content_images_path OR:")
            print("\t\t--content_image content_images")
    elif FLAGS.mode == "train":
        if FLAGS.vgg_path == "vgg19_36.mat":
            net_type = "vgg19"
        elif FLAGS.vgg_path == "vgg16_36.mat":
            net_type = "vgg16"
        else:
            print("Please specify valid vgg net model path.")
            return
            
        train(net_type)

main()