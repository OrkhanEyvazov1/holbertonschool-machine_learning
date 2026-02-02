#!/usr/bin/env python3
''' poisson d '''


class Poisson:
    '''class of poisson d '''
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        ''' init pack '''
        self.lambtha = float(lambtha)

        if data is None:
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        Args:
            k: number of successes
        Returns: PMF value for k
        """
        k = int(k)
        factorial_k = 1
        if k < 0:
            return 0
        for i in range(1, k + 1):
            factorial_k *= i
        pmf = Poisson.e ** -self.lambtha * self.lambtha ** k / factorial_k
        return pmf

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes
        Args:
            k: number of successes
        Returns: CDF value for k
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            factorial_i = 1
            for j in range(1, i + 1):
                factorial_i *= j
            pmf_i =
            Poisson.e ** -self.lambtha * self.lambtha ** i / factorial_i
            cdf += pmf_i
        return cdf
