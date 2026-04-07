#!/usr/bin/env python3
"""5-hue.py
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    a function that changes the hue of an image
    :param image: is a 3D tf.Tensor containing the image to change the
    hue of
    :param delta: is the value to add to the hue channel
    :return: the hue adjusted image
    """
    return tf.image.adjust_hue(image, delta)
