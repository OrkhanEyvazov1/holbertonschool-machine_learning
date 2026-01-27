#!/usr/bin/env python3
''' determinat of 2x2 or 3x3 '''


def determinant(matrix):
    """Calculates the determinant of a 2x2 or 3x3 matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise TypeError("matrix must be a square matrix")
    if len(matrix) == 1:
        return 1
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    elif len(matrix) == 3:
        return (matrix[0][0] * (matrix[1][1] * matrix[2][2] -
                                matrix[1][2] * matrix[2][1]) -
                matrix[0][1] * (matrix[1][0] * matrix[2][2] -
                                matrix[1][2] * matrix[2][0]) +
                matrix[0][2] * (matrix[1][0] * matrix[2][1] -
                                matrix[1][1] * matrix[2][0]))
    else:
        return None  # None for higher order matrices
