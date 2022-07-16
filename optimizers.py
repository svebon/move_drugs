import numpy as np


class Optimizer:
    best_tuple: tuple = None
    best_O_R: float = 1

    def __init__(self, mat_D, vec_T, mat_P):
        self.mat_D = mat_D
        self.vec_T = vec_T
        self.mat_P = mat_P
        self.N = mat_D.shape[0] * mat_D.shape[1]

    def get_O_R(self, a1: float, a2: float, a3: float) -> float:
        w_D = np.multiply(a1, self.mat_D)
        w_T = np.multiply(a2, self.vec_T)

        S = w_D + w_T + a3

        O_R = np.sqrt((S - self.mat_P) ** 2) / self.N

        return O_R

    # Must set self.best_tuple and self.best_O_R and return [best_tuple best_O_R]
    def optimize(self):
        raise NotImplementedError
