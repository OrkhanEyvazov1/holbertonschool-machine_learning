#!/usr/bin/env python3
"""0-conv_forward.py
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural
    network
    Args:
        A_prev (numpy.ndarray):
        Array of shape (m, h_prev, w_prev, c_prev)
        containing the output of the
        previous layer.
        W (numpy.ndarray):
        Array of shape (kh, kw, c_prev, c_new) containing
        the kernels for the
        convolution.
        b (numpy.ndarray):
        Array of shape (1, 1, 1, c_new) containing
        the biases applied to the
        convolution.
        padding (str): Either 'same' or 'valid', indicating
        the type of
        padding used.
        stride (tuple): Tuple of (sh, sw) containing
        the strides for the
        convolution.
    Returns:
        numpy.ndarray:
        Array of shape (m, h_new, w_new, c_new) containing
        the output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph = pw = 0

    h_new = (h_prev - kh + 2 * ph) // sh + 1
    w_new = (w_prev - kw + 2 * pw) // sw + 1

    output = np.zeros((m, h_new, w_new, c_new))

    A_prev_padded = np.pad(
        A_prev,
        ((0,), (ph,), (pw,), (0,)),
        mode='constant'
    )

    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            output[:, i, j, :] = np.tensordot(
                A_prev_padded[:, h_start:h_end, w_start:w_end, :],
                W,
                axes=([1, 2, 3], [0, 1, 2])
            ) + b.reshape(1, c_new)

    return activation(output)
