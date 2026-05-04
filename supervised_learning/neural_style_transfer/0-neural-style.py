#!/usr/bin/env python3
"""Neural style transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Neural style transfer."""

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize the class."""
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Scale an image such that its pixels values are between 0 and 1."""
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > 512 or w > 512:
            if h >= w:
                new_h = 512
                new_w = int(w * (512 / h))
            else:
                new_w = 512
                new_h = int(h * (512 / w))
            image = tf.image.resize(image, (new_h, new_w),
                                    method=tf.image.ResizeMethod.AREA)
        return image / 255.0
