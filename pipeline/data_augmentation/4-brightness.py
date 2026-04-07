#!/usr/bin/env python3
"""4-brightness.py
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    a function that changes the brightness of an image
    :param image: is a 3D tf.Tensor containing the image to change the
    brightness of
    :param max_delta: is the maximum amount the brightness can be changed
    :return: the brightness adjusted image
    """
    delta = tf.random.uniform([], -max_delta, max_delta)
    return tf.image.adjust_brightness(image, delta)
