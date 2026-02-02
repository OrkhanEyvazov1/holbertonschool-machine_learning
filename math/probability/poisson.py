#!/usr/bin/env python3
''' poisson d '''


class Poisson:
    '''class of poisson d '''
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
