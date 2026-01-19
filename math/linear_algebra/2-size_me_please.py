#!/usr/bin/env python3


'''return shape of matrix'''


def matrix_shape(matrix):
    ''' shape of matrix '''
    if not matrix:
        return [0]

    if isinstance(matrix[0], list):
        return [len(matrix), len(matrix[0])]
    else:
        return [len(matrix)]
