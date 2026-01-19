#!/usr/bin/env python3


'''doc here '''


def matrix_transpose(matrix):
    ''' transpose it '''
    return [[matrix[i][j] for i in range(len(matrix))]
            for j in range(len(matrix[0]))]
