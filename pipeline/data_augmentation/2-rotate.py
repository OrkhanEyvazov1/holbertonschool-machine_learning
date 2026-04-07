#!/usr/bin/env python3
"""0-flip.py
"""
import tensorflow as tf


def flip_image(image):
    """
    a function that flips an image horizontally
    :param image: is a 3D tf.Tensor containing the image to flip
    :return: the flipped image
    """
    return tf.image.rot90(image)
