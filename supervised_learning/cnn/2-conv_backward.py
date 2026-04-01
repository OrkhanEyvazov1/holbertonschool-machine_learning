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

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - h_new + kh - sh) / 2))
        pad_w = int(np.ceil((w_prev * sw - w_new + kw - sw) / 2))
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                     (pad_w, pad_w), (0, 0)), mode="constant")
    elif padding == "valid":
        pad_h = pad_w = 0
        A_prev_pad = A_prev

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        dA_prev_pad = np.zeros_like(a_prev_pad)

        for h in range(h_prev):
            for w in range(w_prev):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    dA_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = dA_prev_pad[pad_h:h_prev + pad_h,
                                              pad_w:w_prev + pad_w, :]
        elif padding == "valid":
            dA_prev[i, :, :, :] = dA_prev_pad

    return dA_prev, dW, db
