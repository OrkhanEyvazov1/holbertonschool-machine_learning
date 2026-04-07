#!/usr/bin/env python3
"""1-crop.py
"""
import tensorflow as tf


def crop_image(image, size):
    """
    a function that crops an image
    :param image: is a 3D tf.Tensor containing the image to crop
    :param size: is a tuple containing the height and width of the crop
    :return: the cropped image
    """
    return tf.image.resize_with_crop_or_pad(image, size[0], size[1])
