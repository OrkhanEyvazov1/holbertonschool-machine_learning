#!/usr/bin/env python3
"""3-convolve_grayscale.py
Contains the function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple grayscale images
                m is the number of images
                h is the height in pixels of the images
                w is the width in pixels of the images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
                kh is the height of the kernel
                kw is the width of the kernel
        padding: Either a tuple of (ph, pw), 'same', or 'valid'
                 'same': performs a same convolution
                 'valid': performs a valid convolution
                 tuple: ph is the padding for the height, pw is the padding for the width
                 image is padded with 0's
        stride: tuple of (sh, sw)
                sh is the stride for the height of the image
                sw is the stride for the width of the image

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = kh - 1
        pw = kw - 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph // 2, (ph + 1) // 2), (pw // 2, (pw + 1) // 2)),
        mode='constant',
        constant_values=0
    )

    padded_h = padded_images.shape[1]
    padded_w = padded_images.shape[2]
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            start_h = i * sh
            start_w = j * sw

            patch = padded_images[:, start_h:start_h + kh, start_w:start_w + kw]
            output[:, i, j] = np.sum(patch * kernel, axis=(1, 2))

    return output
