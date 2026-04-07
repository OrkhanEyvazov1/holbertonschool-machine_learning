#!/usr/bin/env python3
"""3-contrast.py
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    a function that changes the contrast of an image
    :param image: is a 3D tf.Tensor containing the image to change the
    contrast of
    :param lower: is the lower bound of the contrast factor
    :param upper: is the upper bound of the contrast factor
    :return: the contrast adjusted image
    """
    level = tf.random.uniform([], lower, upper)
    return tf.image.adjust_contrast(image, level)
