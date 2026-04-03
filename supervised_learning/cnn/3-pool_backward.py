#!/usr/bin/env python3
"""3-pool_backward.py
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode="max"):
    """
    a function that performs back propagation over a pooling layer of a CNN
    :param dA: is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
    the partial derivatives with respect to the output of the pooling layer
    :param A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
    :param kernel_shape: is a tuple of (kh, kw) containing the size of the
    kernel for pooling
    :param stride: is a tuple of (sh, sw) containing the strides for the
    pooling
    :param mode: is a string that indicates the type of pooling, either max
    or average
    :return: the partial derivatives with respect to the previous layer (
    dA_prev)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for img in range(m):
        A_img = A_prev[img]
        dA_img = dA_prev[img]
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    # corners of the slice
                    row_start = row * sh
                    row_end = row * sh + kh
                    col_start = col * sw
                    col_end = col * sw + kw

                    if mode == "max":
                        slice_A = A_img[row_start:row_end, col_start:col_end,
                                        ch]
                        mask = slice_A == np.max(slice_A)
                        dA_img[row_start:row_end,
                               col_start:col_end,
                               ch] += mask * dA[img, row, col, ch]

                    elif mode == "avg":
                        da = dA[img, row, col, ch]
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        dA_img[row_start:row_end,
                               col_start:col_end,
                               ch] += np.ones(shape) * average

    return dA_prev
