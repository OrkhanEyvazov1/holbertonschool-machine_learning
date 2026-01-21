#!/usr/bin/env python3
''' numpy axis based op '''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    ''' doc here '''
    return np.concatenate((mat1, mat2), axis)

