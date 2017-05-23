"""
Helper module to read and preprocess images.
"""
import tensorflow as tf
import os
mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def preprocess(images, image_size, max_length, if_batch=False):
    shape = tf.shape(images)
    size_t = tf.constant(image_size, tf.float64)
    if if_batch:
        height = tf.cast(shape[1], tf.float64)
        width = tf.cast(shape[2], tf.float64)
    else:
        height = tf.cast(shape[0], tf.float64)
        width = tf.cast(shape[1], tf.float64)
    cond_op = tf.less(width, height) if max_length else tf.less(height, width)

    new_height, new_width = tf.cond(
        cond_op,
        lambda: (size_t, (width * size_t) / height),  # if max_length and height > width
        lambda: ((height * size_t) / width, size_t))  # if max_length and height < width

    resized_image = tf.image.resize_images(images, tf.to_int32(new_height), tf.to_int32(new_width))
    # make the input between [-127, 127]
    normalised_image = resized_image - mean_pixel
    return normalised_image


# max_length: Whether size dictates longest or shortest side. Default longest
def get_image(path, image_size, max_length=True, channels=3):
    if_png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image_ = tf.image.decode_png(img_bytes, channels=channels) if if_png \
        else tf.image.decode_jpeg(img_bytes, channels=channels)
    
    return preprocess(image_, image_size, max_length)


def get_image_frame(path, image_format, image_size, max_length=True, channels=3):
    img_bytes = tf.read_file(path)
    cond_op = tf.equal(image_format, "png")
    image_ = tf.cond(
        cond_op,
        lambda: tf.image.decode_png(img_bytes, channels=channels),
        lambda: tf.image.decode_jpeg(img_bytes, channels=channels))

    return preprocess(image_, image_size, max_length)


def get_batch_images(file_names, image_format, image_size, batch_size, max_length=True):
    """get a batch of images, the size of these images should have same size"""
    files = tf.split(0, batch_size, file_names)
    img_bytes = [tf.read_file(_file[0]) for _file in files]

    if image_format == "png":
        decode_image = tf.image.decode_png
    else:
        decode_image = tf.image.decode_jpeg

    # images = [ decode_image(img_b, channels=3) for img_b in img_bytes ]
    # ims_preprocessed = [ preprocess(image, image_size, max_length) for image in images ]
    # return tf.pack(ims_preprocessed)

    images = [decode_image(img_b, channels=3) for img_b in img_bytes]
    return preprocess(tf.pack(images), image_size, max_length, if_batch=True)


def image(n, size, path, epochs=2, shuffle=True, crop=True):
    """Get a batch of images from path for training
    Args:
        n: batch size
        size: image size
        path: training images path
        epochs: training epochs
        shuffle: whether shuffle images
        crop: whether crop images
    Return:
        A batch of images with `size` in `path`
    """
    file_names = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if not shuffle:
        file_names = sorted(file_names)

    png = file_names[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(file_names, num_epochs=epochs, shuffle=shuffle)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    img = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess(img, size, False)
    if not crop:
        return tf.train.batch([processed_image], n, dynamic_pad=True)

    cropped_image = tf.slice(processed_image, [0, 0, 0], [size, size, 3])
    cropped_image.set_shape((size, size, 3))

    images = tf.train.batch([cropped_image], n, num_threads=8)
    return images
