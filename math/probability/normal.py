#!/usr/bin/env python3
'''
Docstring for math tasks
'''


class Normal:
    '''
    Docstring for Normal
    '''
    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        Args:
            x: x-value
        Returns: z-score of x
        """
        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """ Calculates the x-value of a given z-score
        Args:
            z: z-score
        Returns: x-value of z
        """
        x = z * self.stddev + self.mean
        return x

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        Args:
            x: x-value
        Returns: PDF value for x
        """
        coefficient = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        pdf = coefficient * Normal.e ** exponent
        return pdf

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value
        Args:
            x: x-value
        Returns: CDF value for x
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf_approx = (z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 +
                      (z ** 9) / 216)
        cdf = 0.5 * (1 + (2 / (Normal.pi ** 0.5)) * erf_approx)
        return cdf
