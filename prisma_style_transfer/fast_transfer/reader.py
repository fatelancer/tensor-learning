"""
Helper module to read and preprocess images.
"""
import tensorflow as tf

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

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

    new_height, new_width = tf.cond(cond_op,
        lambda: (size_t, (width * size_t) / height), # if max_length and height > width
        lambda: ((height * size_t) / width, size_t)) # if max_length and height < width

    resized_image = tf.image.resize_images(images, tf.to_int32(new_height), tf.to_int32(new_width))
    # make the input bewteen [-127, 127]
    normalised_image = resized_image - mean_pixel
    return normalised_image

# max_length: Wether size dictates longest or shortest side. Default longest
def get_image(path, image_size, max_length=True):
    if_png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if if_png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess(image, image_size, max_length)

def get_batch_images(file_names, format, image_size, batch_size, max_length=True):
    """get a batch of images, the size of these images should have same size"""
    files = tf.split(0, batch_size, file_names)
    img_bytes = [ tf.read_file(file[0]) for file in files ]

    if format == "png":
        decode_image = tf.image.decode_png
    else:
        decode_image = tf.image.decode_jpeg

    # images = [ decode_image(img_b, channels=3) for img_b in img_bytes ]
    # ims_preprocessed = [ preprocess(image, image_size, max_length) for image in images ]
    # return tf.pack(ims_preprocessed)

    images = [ decode_image(img_b, channels=3) for img_b in img_bytes ]
    return preprocess(tf.stack(images), image_size, max_length, if_batch=True)
