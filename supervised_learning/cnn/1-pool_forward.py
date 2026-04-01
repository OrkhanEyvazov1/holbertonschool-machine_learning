#!/usr/bin/env python3
"""1-pool_forward.py
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network
    Args:
        A_prev (numpy.ndarray):
        Array of shape (m, h_prev, w_prev, c_prev)
        containing the output of the
        previous layer.
        kernel_shape (tuple): Tuple of (kh, kw) containing
        the size of the kernel for
        pooling.
        stride (tuple): Tuple of (sh, sw) containing
        the strides for the
        pooling.
        mode (str): Indicates the type of pooling,
        either 'max' or 'avg'.
    Returns:
        numpy.ndarray:
        Array of shape (m, h_new, w_new, c_prev) containing
        the output of the pooling layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_new, w_new, c_prev))

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            if mode == 'max':
                output[:, i, j] = np.max(
                    A_prev[:, h_start:h_end, w_start:w_end],
                    axis=(1, 2)
                )
            else:
                output[:, i, j] = np.mean(
                    A_prev[:, h_start:h_end, w_start:w_end],
                    axis=(1, 2)
                )

    return output
