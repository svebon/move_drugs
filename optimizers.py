import numpy as np
from generator import RandomGenerator
from tqdm import tqdm
from skopt.space import Real
from skopt import gp_minimize
from scipy.optimize import basinhopping


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


class GPOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3, n_jobs=-1, min_improvement=0.01):
        super().__init__(mat_D, vec_T, mat_P)
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size
        self.n_jobs = n_jobs
        self.min_improvement = min_improvement

    def optimize(self):
        space = self.get_space()
        result = gp_minimize(self.get_O_R, space, n_calls=self.n_tuples, n_jobs=self.n_jobs, xi=self.min_improvement)

        self.best_O_R = result.fun
        self.best_tuple = result.x

        return {'best_tuple': self.best_tuple, 'best_O_R': self.best_O_R}

    def get_space(self):
        return [Real(low=0, high=1)] * self.tuples_size


class BHOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3, guess: tuple = (0.5, 0.5, 0.5),
                 timeout=1000, min_imp=0.01, min_imp_timeout=100):
        super().__init__(mat_D, vec_T, mat_P)
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size
        self.guess = guess
        self.timeout = timeout
        self.min_imp = min_imp
        self.min_imp_timeout = min_imp_timeout

    def optimize(self):
        result = basinhopping(self.get_O_R, self.guess, niter=self.n_tuples, niter_success=self.timeout)

        self.best_O_R = result.fun
        self.best_tuple = result.x

        return {'best_tuple': self.best_tuple, 'best_O_R': self.best_O_R}

    def save_minimum(self, x, O_R, accepted):
        if not accepted or not O_R < self.best_O_R:
            return None

        positive = self.positive_alphas(x)
        if positive:
            old_best_O_R = self.best_O_R
            self.best_O_R = O_R
            self.best_tuple = x

            if old_best_O_R - O_R < self.min_imp:
                self.min_imp_timeout -= 1

                return self.min_imp_timeout < 0

    @staticmethod
    def positive_alphas(alphas):
        for alpha in alphas:
            if alpha < 0:
                return False
        return True
