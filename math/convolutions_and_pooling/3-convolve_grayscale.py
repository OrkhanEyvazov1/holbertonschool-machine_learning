#!/usr/bin/env python3
"""3-convolve_grayscale.py
Contains the function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grayscale images with custom padding and stride

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: number of images
            h: height in pixels of the images
            w: width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for
                the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding: either a tuple of (ph, pw), 'same', or 'valid'
            if 'same', performs a same convolution
            if 'valid', performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh is the stride for the height of the image
            sw is the stride for the width of the image
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = ((h - 1) * sh + kh - h) // 2
        pad_w = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    out_h = (h + 2 * pad_h - kh) // sh + 1
    out_w = (w + 2 * pad_w - kw) // sw + 1
    convolved_images = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            convolved_images[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved_images
