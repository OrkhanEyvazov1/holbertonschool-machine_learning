#!/usr/bin/env python3


''' poly of derv '''


def poly_derivative(poly):
    '''poly of dev doc'''
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    if len(poly) == 1:
        return [0]    
    derivative = []
    for power in range(1, len(poly)):
        new_coef = power * poly[power]
        derivative.append(new_coef)

    return derivative
