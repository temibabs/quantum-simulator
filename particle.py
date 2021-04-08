import numpy as np

from util import generate_random


class QuantumParticle(object):
    def __init__(self, dimensions=3):
        self.amplitudes = generate_random(dimensions)
        self.state = self.amplitudes
        self.orientation = np.random.random(dimensions)

        self.kets = self.amplitudes
        self.bras = np.conjugate(self.amplitudes)

    def initialize(self):
        pass

    def evolve(self, hermitian):
        print()

    def measurePosition(self, dimension: int):
        p_one = self.bras[0, 0] * self.kets[0, 0]
        p_minus_one = self.bras[dimension, 1] * self.kets[dimension, 1]
        return np.random.choice([+1, -1], p=[p_one, p_minus_one])

    def entangle(self):
        print()

    def __str__(self):
        return f'\n====================' \
               f'\n{QuantumParticle.__name__}: ' \
               f'\n\namplitudes=\n{self.amplitudes}' \
               f'\n\norientation=\n{self.orientation}' \
               f'\n===================='
