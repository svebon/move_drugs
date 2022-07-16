import numpy as np
from generator import RandomGenerator
from tqdm import tqdm


class Optimizer:
    best_tuple: tuple = None
    best_O_R: float = 1

    def __init__(self, mat_D, vec_T, mat_P):
        self.mat_D = mat_D
        self.vec_T = vec_T
        self.mat_P = mat_P
        self.N = mat_D.shape[0] * mat_D.shape[1]

    def get_O_R(self, alphas) -> float:
        a1, a2, a3 = alphas
        w_D = np.multiply(a1, self.mat_D)
        w_T = np.multiply(a2, self.vec_T)

        S = w_D + w_T + a3

        O_R = np.sqrt((S - self.mat_P) ** 2) / self.N

        return O_R

    # Must set self.best_tuple and self.best_O_R and return {best_tuple: self.best_tuple, best_O_R: self.best_O_R}
    def optimize(self):
        raise NotImplementedError


class RandomOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3):
        super().__init__(mat_D, vec_T, mat_P)
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size

    def optimize(self):
        tuples = RandomGenerator(self.n_tuples, self.tuples_size)

        with tqdm(tuples, desc='Testing tuples', total=self.n_tuples) as pbar:
            for t in pbar:
                O_R = self.get_O_R(t)

                pbar.set_postfix(O_R=O_R, alphas=t)

                if O_R < self.best_O_R:
                    self.best_tuple = t
                    self.best_O_R = O_R

        return {'best_tuple': self.best_tuple, 'best_O_R': self.best_O_R}
