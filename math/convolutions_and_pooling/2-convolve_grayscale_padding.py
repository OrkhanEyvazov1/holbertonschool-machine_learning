#!/usr/bin/env python3
"""
2-convolve_grayscale_padding.py
Contains the function convolve_grayscale_padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
        padding: tuple of (pad_h, pad_w) containing the padding for the convolution
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')
                           
    out_h = h + (2 * pad_h) - kh + 1
    out_w = w + (2 * pad_w) - kw + 1
    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded_images[:, i:i + kh, j:j + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved_images
