#!/usr/bin/env python3
''' multi multi.. '''


def determinant(matrix):
    """
    calculates the determinant of a matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Determinant of matrix
    """
    mat_l = len(matrix)
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(mat, list) for mat in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix[0] and mat_l != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if mat_l == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a square matrix")

    return multi_determinant(matrix)


def minor_val(matrix, idx_r, idx_c):
    """
    function that computes minor in each idx position of the given matrix
    Args:
        matrix: given matrix
        idx_r: row skipped
        idx_c: col skipped
    Returns: determinant of the matrix with row and col skipped
    """
    minor_mat = [rows[:idx_c] + rows[idx_c + 1:]
                 for rows in (matrix[:idx_r] + matrix[idx_r + 1:])]
    return determinant(minor_mat)


def minor(matrix):
    """
    Compute the minor of a given matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Minor of a matrix
    """
    mat_l = len(matrix)
    range_mat_l = range(len(matrix))

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(mat, list) for mat in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(mat) == mat_l for mat in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1:
        return [[1]]

    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            minor_r.append(minor_c)
        minor_values.append(minor_r)
    return minor_values
