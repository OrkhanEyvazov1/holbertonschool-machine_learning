#!/usr/bin/env python3
'''
Docstring for math tasks
'''


class Exponential:
    '''
    Docstring for Exponential
    '''
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)

        if data is None:
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        Args:
            x: time period
        Returns: PDF value for x
        """
        if x < 0:
            return 0
        pdf = self.lambtha * Exponential.e ** (-self.lambtha * x)
        return pdf

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period
        Args:
            x: time period
        Returns: CDF value for x
        """
        if x < 0:
            return 0
        cdf = 1 - Exponential.e ** (-self.lambtha * x)
        return cdf
