#!/usr/bin/env python3
''' matrix mult '''


def mat_mul(mat1, mat2):
    ''' doc here '''
    if len(mat1[0]) != len(mat2):
        return None
    else:
        mat2_t = list(zip(*mat2))
        return [
            [sum(a * b for a, b in zip(row_a, col_b)) for col_b in mat2_t]
            for row_a in mat1
        ]
