from random import uniform
import numpy as np


class RandomGenerator:
    """Iterator that generates random tuples of size tuple_size, n_tuple times"""
    def __init__(self, n_tuple: int, tuple_size: int = 3):
        if n_tuple < 1:
            raise ValueError('n_tuple must be > 0')
        self.n_tuple = n_tuple
        self.tuple_size = tuple_size

    def __iter__(self):
        return self

    def __next__(self):
        self.n_tuple -= 1
        if self.n_tuple < 0:
            raise StopIteration

        return np.random.rand(self.tuple_size)
