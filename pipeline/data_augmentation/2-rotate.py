#!/usr/bin/env python3
"""0-flip.py
"""
import tensorflow as tf


def rotate_image(image):
    """
    a function that rotates an image by 90 degrees
    :param image: is a 3D tf.Tensor containing the image to rotate
    :return: the rotated image
    """
    return tf.image.rot90(image)
