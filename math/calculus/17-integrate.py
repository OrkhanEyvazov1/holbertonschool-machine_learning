#!/usr/bin/env python3


'''poly of integral'''


def poly_integral(poly):
    """Calculate the integral of a polynomial."""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None
    integral = [0.0]
    for i in range(len(poly)):
        new_coeff = poly[i] / (i + 1)
        integral.append(new_coeff)

    return integral
