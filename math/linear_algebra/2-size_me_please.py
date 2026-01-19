#!/usr/bin/env python3


'''return shape of matrix'''


def matrix_shape(matrix):
    ''' shape of matrix '''
    shape = []
    current_layer = matrix
    while isinstance(current_layer, list):
        shape.append(len(current_layer))
        if len(current_layer) == 0:
            break
        current_layer = current_layer[0]
    return shape
