#!/usr/bin/env python3
"""2-conv_backward.py
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding='same', stride=(1, 1)):
    """Performs back propagation over a convolutional layer of a neural
    network
    Args:
        dZ (numpy.ndarray):
        Array of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the unactivated
        output of the convolutional layer.
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
        dA_prev (numpy.ndarray):
            Array of shape (m, h_prev, w_prev, c_prev)
            containing the partial derivatives with respect to
            the output of the previous layer.
        dW (numpy.ndarray):
            Array of shape (kh, kw, c_prev, c_new) containing
            the partial derivatives with respect to the kernels
            of the convolution.
        db (numpy.ndarray):
            Array of shape (1, 1, 1, c_new) containing
            the partial derivatives with respect to the biases
            of the convolution.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph = pw = 0
        A_prev_pad = A_prev

    h_new = (h_prev - kh + 2 * ph) // sh + 1
    w_new = (w_prev - kw + 2 * pw) // sw + 1

    dA_prev = np.zeros((m, h_prev + 2 * ph, w_prev + 2 * pw, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))
    A_prev_padded = np.pad(
        A_prev,
        ((0,), (ph,), (pw,), (0,)),
        mode='constant'
    )
    dA_prev_padded = np.pad(
        dA_prev,
        ((0,), (ph,), (pw,), (0,)),
        mode='constant'
    )
    for i in range(h_new):
        for j in range(w_new):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            dA_prev_padded[:, h_start:h_end, w_start:w_end, :] += np.tensordot(
                dZ[:, i, j, :],
                W,
                axes=([1], [3])
            )
            dW += np.tensordot(
                A_prev_padded[:, h_start:h_end, w_start:w_end, :],
                dZ[:, i, j, :],
                axes=([0], [0])
            )
            db += np.sum(dZ[:, i, j, :], axis=0).reshape(1, 1, 1, c_new)

    dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
