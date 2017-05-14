"""Style transfer.
This file is used to define transfer network architecture.

"""
import tensorflow as tf


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT', norm="batch"):
    """
    Define a convolution layer with input x.
    Args:
        x: `tf.Tensor`, input of conv layer.
        input_filters: input channels.
        output_filters: output channels.
        kernel: width or height of kernel.
        strides: conv strides.
        mode: padding mode, "REFLECT" or "CONSTANT" or "SYMMETRIC".
        norm: norm type, "batch" or "instance" or "no".

    Return:
        A `Tensor`, output of the layer
    """
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        # weight = _variable_with_weight_decay(shape, 0.1, wd)
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        # Add pad
        x_padded = tf.pad(x, [[0, 0], [kernel / 2, kernel / 2], [kernel / 2, kernel / 2], [0, 0]], mode=mode)
        convolved = tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding="VALID", name='conv')

        if norm == "batch":
            normalized = batch_norm(convolved, output_filters)
            return normalized
        elif norm == "instance":
            normalized = batch_norm(convolved, output_filters, norm_shape=(1, 2))
            return normalized
        else:
            return convolved


def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('conv_transpose') as scope:
        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.pack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1],
                                           padding=padding, name='conv_transpose')

        normalized = batch_norm(convolved, output_filters)
        return normalized


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training=True):
    with tf.variable_scope('conv_transpose') as scope:
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, new_height, new_width, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # shape = [kernel, kernel, input_filters, output_filters]
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def batch_norm(x, size, norm_shape=(0, 1, 2)):
    batch_mean, batch_var = tf.nn.moments(x, norm_shape, keep_dims=True)
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch')


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(var, epsilon)))


def residual(x, filters, kernel, strides):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides)

        residual_ = x + conv2

        return residual_


def net(image, if_train=True, input_channels=3):
    # Add border to reduce border effects
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
        deconv1 = tf.nn.relu(resize_conv2d(res5, 128, 64, 3, 2, training=if_train))
    with tf.variable_scope('deconv2'):
        deconv2 = tf.nn.relu(resize_conv2d(deconv1, 64, 32, 3, 2, training=if_train))
    with tf.variable_scope('conv4'):
        deconv3 = tf.nn.tanh(conv2d(deconv2, 32, 3, 9, 1, norm="instance"))

    y = deconv3 * 127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.pack([-1, height - 20, width - 20, -1]))

    return y
