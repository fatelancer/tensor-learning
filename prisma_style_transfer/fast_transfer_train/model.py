"""Style transfer.
This file is used to define transfer network architecture.

"""
import tensorflow as tf
import reader
import vgg


FLAGS = tf.app.flags.FLAGS

def conv2d(x, input_channels, output_channels, kernel, strides, mode='REFLECT', norm="batch", if_norm=True):
    """
    Define a convolution layer with input x.
    Args:
        x: `tf.Tensor`, input of conv layer.
        input_channels: input channels.
        output_channels: output channels.
        kernel: width or height of kernel.
        strides: conv strides.
        mode: padding mode, "REFLECT" or "CONSTANT" or "SYMMETRIC".
        norm: norm type, "batch" or "instance" or "no".

    Return:
        A `Tensor`, output of the layer
    """
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_channels, output_channels]
        # weight = _variable_with_weight_decay(shape, 0.1, wd)
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        # Add pad
        x_padded = tf.pad(x, [[0, 0], [kernel // 2, kernel // 2], [kernel // 2, kernel // 2], [0, 0]], mode=mode)
        convolved = tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding="VALID", name='conv')

        if not if_norm:
            return convolved

        if norm == "batch":
            normalized = normalize(convolved, output_channels, norm_type='batch')
            return normalized
        elif norm == "instance":
            normalized = normalize(convolved, output_channels, norm_type='instance')
            return normalized
        else:
            return convolved


def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME', if_norm=True):
    with tf.variable_scope('conv_transpose') as scope:
        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1],
                                           padding=padding, name='conv_transpose')

        if not if_norm:
            return convolved
        normalized = normalize(convolved, output_filters, norm_type='batch')
        return normalized

# 这个用给 super-resolution的,用不到
def resize_conv2d(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose') as scope:

        # 这两种有什么区别么？ 在没开始训练之前
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, (tf.to_int32(new_height), tf.to_int32(new_width)), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)

# Input is a 4-D Tensor.
def normalize(x, size, norm_type='batch'):
    if norm_type not in ['batch', 'instance']:
        raise Exception("NO matching type of normalization!")

    norm_shape = (0,1,2) if norm_type == 'batch' else (1,2)

    mean, var = tf.nn.moments(x, norm_shape, keep_dims=True)


    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, mean, var, beta, scale, epsilon, name='batch')

def ada_norm(image_features, style_features):
    # 第一个batch维度肯定是1
    i_mean, i_var = tf.nn.moments(image_features, (0,1,2), keep_dims=True)

    s_mean, s_var = tf.nn.moments(style_features, (0,1,2), keep_dims=False)
    epsilon = 1e-3
    return tf.nn.batch_normalization(image_features, i_mean, i_var, s_mean, s_var, epsilon, name="adain_norm")


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
            features.append(gram(net[layer], FLAGS.batch_size))

        with tf.Session() as sess:
            return sess.run(features)



def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides)

        residual_ = x + conv2

        return residual_

# 定义前向生成网络
def net(image, if_train=True, input_channels=3):
    # Add border to reduce border effects
    # 为了使用 REFLECT 的 填充方式 才加入了 pad
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = tf.nn.relu(conv2d(image, input_channels, 32, 9, 1))
    with tf.variable_scope('conv2'):
        conv2 = tf.nn.relu(conv2d(conv1, 32, 64, 3, 2))
    with tf.variable_scope('conv3'):
        conv3 = tf.nn.relu(conv2d(conv2, 64, 128, 3, 2))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        ### deconv1 = tf.nn.relu(resize_conv2d(res5, 128, 64, 3, 2, training=if_train))
        deconv1 = tf.nn.relu(conv2d_transpose(res5, 128, 64, 3, 2))
    with tf.variable_scope('deconv2'):
        ### deconv2 = tf.nn.relu(resize_conv2d(deconv1, 64, 32, 3, 2, training=if_train))
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 32, 3, 2))
    with tf.variable_scope('conv4'):
        # Use a scaled tanh to ensure the output image pixels in [0, 255]
        deconv3 = tf.nn.tanh(conv2d(deconv2, 32, 3, 9, 1, norm="instance"))

    y = deconv3 * 127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y


# Done
def gram(layer, factor):
    """ Get style with gram matrix.
    layer with shape(batch, height, weight, channels) of activations.
    """
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(tf.matrix_transpose(filters), filters) / tf.to_float(size / factor) # FLAGS.batch_size)

    return grams


# 定义前向网络, 使用 Instance Normalization, 并且放在激活函数的后面.
def net2(image, if_train=True, input_channels=3):
    # Add border to reduce border effects
    # 为了使用 REFLECT 的 填充方式 才加入了 pad
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv1'):
        conv1 = normalize(tf.nn.relu(conv2d(image, input_channels, 32, 9, 1, if_norm=False)), 32, norm_type='instance')
    with tf.variable_scope('conv2'):
        conv2 = normalize(tf.nn.relu(conv2d(conv1, 32, 64, 3, 2, if_norm=False)), 64, norm_type='instance')
    with tf.variable_scope('conv3'):
        conv3 = normalize(tf.nn.relu(conv2d(conv2, 64, 128, 3, 2, if_norm=False)), 128, norm_type='instance')
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        ### deconv1 = tf.nn.relu(resize_conv2d(res5, 128, 64, 3, 2, training=if_train))
        deconv1 = normalize(tf.nn.relu(conv2d_transpose(res5, 128, 64, 3, 2, if_norm=False)), 64, norm_type='instance')
    with tf.variable_scope('deconv2'):
        ### deconv2 = tf.nn.relu(resize_conv2d(deconv1, 64, 32, 3, 2, training=if_train))
        deconv2 = normalize(tf.nn.relu(conv2d_transpose(deconv1, 64, 32, 3, 2)), 32, norm_type='instance')
    with tf.variable_scope('conv4'):
        # Use a scaled tanh to ensure the output image pixels in [0, 255]
        # [-1, 1]
        deconv3 = tf.nn.tanh(conv2d(deconv2, 32, 3, 9, 1, if_norm=False))

    # y need add reader.norm
    # 这里 * 127.5 的原因是前面已经执行了 Normalization把 [0,1] 转为了 0 为中心
    # 并且后面的输出结果会 加上 mean_pixel, 然后再截取回 [0, 255]
    y = deconv3 * 127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y




# 定义Adaptive网络结构的前向网络部分
# Image and Style_image is a 4-D Tensor.
def AdaIn_net(image, style_image, input_channels=3):
    net, _ = vgg.net(FLAGS.vgg_path, tf.concat([image, style_image], 0), "vgg19")

    image_features, style_features = tf.split(net['relu4_1'], num_or_size_splits=2, axis=0)
    with tf.variable_scope("AdaIn"):
        ada_out = ada_norm(image_features, style_features)

    with tf.variable_scope("d_conv4_1"):
        d_conv4_1 = tf.nn.relu(conv2d(image_features, 512, 256, 3, 1, if_norm=False))

    with tf.variable_scope("pooling_3"):
        d_pooling_3 = resize_conv2d(d_conv4_1, 256, 256, 3, 2)

    with tf.variable_scope("d_conv3_4"):
        d_conv3_4 = tf.nn.relu(conv2d(d_pooling_3, 256, 256, 3, 1, if_norm=False))

    with tf.variable_scope("d_conv3_3"):
        d_conv3_3 = tf.nn.relu(conv2d(d_conv3_4, 256, 256, 3, 1, if_norm=False))

    with tf.variable_scope("d_conv3_2"):
        d_conv3_2 = tf.nn.relu(conv2d(d_conv3_3, 256, 256, 3, 1, if_norm=False))

    with tf.variable_scope("d_conv3_1"):
        d_conv3_1 = tf.nn.relu(conv2d(d_conv3_2, 256, 128, 3, 1, if_norm=False))

    with tf.variable_scope("pooling_2"):
        d_pooling_2 = resize_conv2d(d_conv3_1, 128, 128, 3, 2)

    with tf.variable_scope("d_conv2_2"):
        d_conv2_2 = tf.nn.relu(conv2d(d_pooling_2, 128, 128, 3, 1, if_norm=False))

    with tf.variable_scope("d_conv2_1"):
        d_conv2_1 = tf.nn.relu(conv2d(d_conv2_2, 128, 64, 3, 1, if_norm=False))

    with tf.variable_scope("pooling_1"):
        d_pooling_1 = resize_conv2d(d_conv2_1, 64, 64, 3, 2)

    with tf.variable_scope("d_conv1_2"):
        d_conv1_2 = tf.nn.relu(conv2d(d_pooling_1, 64, 64, 3, 1, if_norm=False))

    with tf.variable_scope("d_conv1_1"):
        d_conv1_1 = tf.nn.relu(conv2d(d_conv1_2, 64, 3, 3, 1, if_norm=False))


    return d_conv1_1, image_features