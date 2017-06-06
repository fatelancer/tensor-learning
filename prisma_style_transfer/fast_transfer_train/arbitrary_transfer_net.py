'''
Arbitrary style transfer

Based on Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization, Xun Huang.

Author: Liu Yi

'''

import tensorflow as tf

import reader
import model
import vgg
import time
import utils
import os

FLAGS = tf.app.flags.FLAGS
image = None

def AdaIn_loss(style_path):
    """Compute perceptual loss of content and style

    Return:
        generated 前向生成网络
        images 输入图片(batch based)
        loss 各种loss.
    """
    global image

    # Set style layers and content layers in vgg net
    content_layers = FLAGS.content_layers.split(',')


    style_layers = FLAGS.style_layers.split(',')
    style_features_t = model.get_style_features(style_path, style_layers, "vgg19")

    style_image = reader.get_image(style_path, FLAGS.image_size)

    # Read images from dataset
    # 这里使用多组没有意义，就只用一张照片即可
    if not image:
        image = reader.image(1, FLAGS.image_size, FLAGS.train_images_path, epochs=FLAGS.epoch)

    # Transfer images
    # Ada_mid is used to calculate content loss.
    generated, Ada_features = model.AdaIn_net(image, style_image)

    # Process generated and original images with vgg
    net, _ = vgg.net(FLAGS.vgg_path, tf.concat([generated, image], 0), "vgg19")

    # Get content loss
    content_loss = 0
    for layer in content_layers:
        # 平均分为两组，每组都是batch长度的图片组
        gen_features, image_features = tf.split(net[layer], num_or_size_splits=2, axis=0)
        size = tf.size(image_features)
        content_loss += tf.nn.l2_loss(Ada_features - image_features) / tf.to_float(size)
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
    total_v_loss = model.total_variation_loss(generated)
    loss = FLAGS.style_weight * style_loss + FLAGS.content_weight * content_loss + FLAGS.tv_weight * total_v_loss

    return generated, image, content_loss, style_loss, total_v_loss, loss




def train():
    """ training a model for style transfer. """
    # Record training start time
    train_start = time.time()

    model_suffix = utils.get_ar_model_suffix()
    model_path = os.path.join("models", "ar_" + FLAGS.model_name + model_suffix)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_name = os.path.join(model_path, FLAGS.model_name)

    # Summary path
    summary_path = os.path.join(FLAGS.summary_path, "ar_" + FLAGS.model_name + model_suffix)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # Record running configs in log file
    utils.log_train_configs(train_start, model_name, summary_path)



    # 每张风格图训练 style_loop 次，然后保存模型换下一张，重新加载模型然后训练
    # 一共训练 style_epoch 次
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
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

        # 使用一组风格图进行轮换训练
        style_images = []
        if not os.path.isdir(FLAGS.style_path):
            print("Please Input an valid style_paths: It must be a directory.")
        for file in os.listdir(FLAGS.style_path):
            style_images.append(os.path.join(FLAGS.style_path, file))


        try:
            for i in range(FLAGS.style_epoch):
                for style_image in style_images:
                    generated, image, content_loss, style_loss, total_v_loss, loss = AdaIn_loss(style_image)

                    # Add summary
                    content_loss_summary = utils.scalar_variable_summaries(content_loss, "content_loss")
                    style_loss_summary = utils.scalar_variable_summaries(style_loss, "style_loss")
                    tv_loss_summary = utils.scalar_variable_summaries(total_v_loss, "total_variation_loss")
                    loss_summary = utils.scalar_variable_summaries(loss, "TOTAL_LOSS")

                    output_format = tf.saturate_cast(tf.concat([generated, image], 0) + reader.mean_pixel, tf.uint8)
                    # output_images = tf.saturate_cast(generated + reader.mean_pixel, tf.uint8)

                    # Add output image summary
                    image_summary = tf.summary.image("output-image", output_format, max_outputs=8)

                    merge_summary = tf.summary.merge(
                        content_loss_summary + style_loss_summary + tv_loss_summary + loss_summary + [image_summary])


                    global_step = tf.Variable(step, name="global_step", trainable=False)
                    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, global_step=global_step)

                    for j in range(FLAGS.style_loop):
                        _, c_loss, s_loss, tv_loss, total_loss, step = sess.run(
                            [train_op, content_loss, style_loss, total_v_loss, loss, global_step])

                        if step % FLAGS.record_interval == 0:
                            elapsed_time = time.time() - start_time
                            total_time += elapsed_time
                            start_time = time.time()
                            summary = sess.run(merge_summary)
                            # Record summaries
                            summary_writer.add_summary(summary, step)

                         #   if total_loss < best_loss:
                                # im_summary = sess.run(im_merge)
                                # train_writer.add_summary(im_summary, step)
                                # Save checkpoint file
                        #       best_loss = total_loss
                    saver.save(sess, model_name, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Finished training -- epoch limit reached!')

        except KeyboardInterrupt:
            print("Terminated by Keyboard Interrupt")

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




def gen_single():
    pass

def gen_from_directory():
    pass


