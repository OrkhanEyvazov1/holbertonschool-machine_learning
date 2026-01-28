#!/usr/bin/env python3
''' inverse 1/det * ct '''


def multi_determinant(matrix):
    """
    Helper function to compute determinant of matrix larger than 2x2
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Determinant of matrix
    """
    det = 0
    for c in range(len(matrix)):
        sub_matrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)
    return det


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
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix == [[]]:
        return 1
    if mat_l == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

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


def cofactor(matrix):
    """
    Compute the cofactor of a given matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Cofactor of a matrix
    """
    matrix_minors = minor(matrix)
    cofactor_matrix = []
    for r in range(len(matrix_minors)):
        cofactor_row = []
        for c in range(len(matrix_minors)):
            cofactor_val = ((-1) ** (r + c)) * matrix_minors[r][c]
            cofactor_row.append(cofactor_val)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def matrix_transpose(matrix):
    ''' transpose it '''
    for i in range(len(matrix)):
        for j in range(i, len(matrix[0])):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


def adjugate(matrix):
    """
    Compute the adjugate of a given matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Adjugate of a matrix
    """
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = matrix_transpose(cofactor_matrix)
    return adjugate_matrix


def inverse(matrix):
    """
    Compute the inverse of a given matrix
    Args:       matrix: list of lists whose determinant should be calculated
    Returns: Inverse of a matrix
    """
    det = determinant(matrix)
    if det == 0:
        raise ValueError("matrix is singular and cannot be inverted")
    adjugate_matrix = adjugate(matrix)
    inverse_matrix = []
    for r in range(len(adjugate_matrix)):
        inverse_row = []
        for c in range(len(adjugate_matrix)):
            inverse_val = adjugate_matrix[r][c] / det
            inverse_row.append(inverse_val)
        inverse_matrix.append(inverse_row)
    return inverse_matrix
