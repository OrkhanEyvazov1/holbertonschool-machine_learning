#!/usr/bin/env python3
"""3-convolve_grayscale.py
Contains the function convolve_grayscale
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.
    
    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)
    
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    # Determine padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    
    # Pad images
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    
    # Calculate output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1
    
    # Initialize output array
    output = np.zeros((m, out_h, out_w))
    
    # Perform convolution with only two loops (i and j)
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region for convolution
            h_start = i * sh
            w_start = j * sw
            region = padded_images[:, h_start:h_start + kh, w_start:w_start + kw]
            
            # Apply kernel (element-wise multiplication and sum)
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    
    return output
