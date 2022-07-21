import numpy as np
from generator import RandomGenerator
from tqdm import tqdm
from skopt.space import Real
from skopt import gp_minimize
from scipy.optimize import basinhopping
import warnings
import dataclasses
from scipy.optimize import OptimizeResult


@dataclasses.dataclass
class BestResult:
    _alphas: np.ndarray
    _O_R: np.ndarray
    _O_R_avg: float

    def update(self, result: OptimizeResult = None, new_alphas: np.ndarray = None, new_O_R_avg: float = None,
               new_O_R: np.ndarray = None):
        if not new_O_R:
            raise ValueError('new_O_R must be provided')

        if result is not None:
            new_alphas = result.x
            new_O_R_avg = result.fun

        self._alphas = new_alphas
        self._O_R = new_O_R
        self._O_R_avg = new_O_R_avg

    @property
    def alphas(self):
        return self._alphas

    @alphas.setter
    def alphas(self, new_alphas: np.ndarray):
        if not isinstance(new_alphas, np.ndarray):
            raise TypeError('new_alphas must be a numpy array')

        if new_alphas.ndim > 1:
            raise ValueError('new_alphas must be a 1D array')

        self._alphas = new_alphas

    @property
    def O_R(self):
        return self._O_R

    @O_R.setter
    def O_R(self, new_O_R: np.ndarray):
        if not isinstance(new_O_R, np.ndarray):
            raise TypeError('new_alphas must be a numpy array')

        if new_O_R.ndim != 2:
            raise ValueError('new_O_R must be a 2D array')

        self._O_R = new_O_R


class Optimizer:
    """
    Optimizer interface
    """
    best_tuple: tuple = None  #: Best tuple of alphas
    best_O_R: np.ndarray  #: Best O_R
    best_O_R_avg: float = 1  #: Best O_R average

    def __init__(self, mat_D: np.ndarray, vec_T: np.ndarray, mat_P: np.ndarray, n_tuples: int = 1000 * 1000,
                 tuples_size: int = 3, min_imp=0.01, min_imp_timeout=10):
        """
        Parameters
        ----------
        mat_D: np.ndarray
            Drugs-receptors docking scores matrix
        vec_T: np.ndarray
            T vector
        mat_P: np.ndarray
            Drugs-receptors interactions matrix
        n_tuples: int
            Number of tuples to test
        tuples_size: int
            Size of tested tuples
        min_imp: float, default: 0.01
            Minimum improvement to continue optimization
        min_imp_timeout: int, default: 10
            Max number of consecutive iterations with an insufficient improvement before stop optimization
        """
        self.mat_D = mat_D
        self.vec_T = vec_T
        self.mat_P = mat_P
        self.N = mat_D.shape[0] * mat_D.shape[1]
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size
        self.min_imp = min_imp
        self.min_imp_timeout = min_imp_timeout
        self.failed_iterations = 0
        self.pbar = tqdm(desc='Testing tuples', total=self.n_tuples)

    def get_O_R(self, alphas) -> np.ndarray:
        """
        Calculates O_R for given alphas

        Parameters
        ----------
        alphas: list
            List of floats
        Returns
        -------
        numpy.ndarray: O_R value
        """
        a1, a2, a3 = alphas
        w_D = np.multiply(a1, self.mat_D)
        w_T = np.multiply(a2, self.vec_T)

        S = w_D + w_T + a3

        O_R = np.sqrt((S - self.mat_P) ** 2) / self.N

        return O_R

    def better(self, O_R: np.ndarray):
        return self.mat_avg(O_R) < self.best_O_R_avg

    # Must set self.best_tuple and self.best_O_R and return {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}
    def optimize(self):
        """
        Must set self.best_tuple and self.best_O_R and return {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}
        """
        raise NotImplementedError

    @staticmethod
    def mat_avg(mat: np.ndarray) -> float:
        n_elements = mat.shape[0] * mat.shape[1]
        return np.sum(mat) / n_elements

    def get_O_R_avg(self, alphas) -> float:
        """
        Calculates O_R average for given alphas

        Parameters
        ----------
        alphas: list
            List of floats
        Returns
        -------
        float: O_R average
        """
        O_R = self.get_O_R(alphas)
        return self.mat_avg(O_R)


class RandomOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3):
        super().__init__(mat_D, vec_T, mat_P, n_tuples, tuples_size)

    def optimize(self):
        """
        Optimize alphas using random tuples.\n
        On each iteration, a new tuple is generated and tested; if it is better than previous best, it is saved.

        Returns
        -------
        dict: {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}
        """
        tuples = RandomGenerator(self.n_tuples, self.tuples_size)

        with tqdm(tuples, desc='Testing tuples', total=self.n_tuples) as pbar:
            for t in pbar:
                O_R = self.get_O_R(t)

                if self.better(O_R):
                    self.best_tuple = t
                    self.best_O_R = O_R
                    self.best_O_R_avg = self.mat_avg(O_R)

        return {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}


class GPOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3, min_imp=0.01, min_imp_timeout=10,
                 n_jobs=-1):
        """
        Parameters
        ----------
        n_jobs: int, default: -1
            Number of jobs to be used by skopt
        """
        super().__init__(mat_D, vec_T, mat_P, n_tuples=n_tuples, tuples_size=tuples_size,
                         min_imp=min_imp, min_imp_timeout=min_imp_timeout)
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size
        self.n_jobs = n_jobs

    def optimize(self):
        """
        Optimize alphas exploiting Gaussian Processes.

        Returns
        -------
        dict: {'best_tuple': self.best_tuple, 'best_O_R': self.best_O_R}
        """
        space = self.space
        warnings.simplefilter('ignore')
        result = gp_minimize(self.get_O_R_avg, space, n_calls=self.n_tuples, n_jobs=self.n_jobs,
                             callback=self.check_min_imp)
        warnings.simplefilter('default')

        self.best_O_R_avg = result.fun
        self.best_tuple = result.x
        self.best_O_R = self.get_O_R(result.x)

        if self.pbar:
            self.pbar.set_description('Completed')
            self.pbar.close()

        return {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}

    def check_min_imp(self, result) -> bool:
        if result.fun > self.best_O_R_avg:
            self.pbar.update()
            return False

        if self.best_O_R_avg - result.fun < self.min_imp:
            self.failed_iterations += 1
            self.pbar.set_postfix(best_O_R=self.best_O_R_avg, fails=f'{self.failed_iterations}/{self.min_imp_timeout}')

            if self.failed_iterations >= self.min_imp_timeout:
                self.pbar.set_description('Min Improvement timeout reached')
                self.pbar.close()
                return True
        else:
            self.failed_iterations = 0

        self.best_O_R_avg = result.fun
        self.pbar.update()

    @property
    def space(self):
        """
        Generates the space of alphas to be tested.

        Returns
        -------
        list: list of floats in the range [0, 1]
        """
        return [Real(low=0, high=1)] * self.tuples_size


class BHOptimizer(Optimizer):
    def __init__(self, mat_D, vec_T, mat_P, n_tuples=1000 * 1000, tuples_size=3, guess: list = [0.5, 0.5, 0.5],
                 timeout=1000, min_imp=0.01, min_imp_timeout=100):
        """

        Parameters
        ----------
        guess: list, default: [0.5, 0.5, 0.5]
            Initial guess for alphas
        timeout: int, default: 1000
            Maximum number of iterations that can be performed without improvement
        """
        super().__init__(mat_D, vec_T, mat_P, n_tuples=n_tuples, tuples_size=tuples_size,
                         min_imp=min_imp, min_imp_timeout=min_imp_timeout)
        self.n_tuples = n_tuples
        self.tuples_size = tuples_size
        self.guess = guess
        self.timeout = timeout
        self.min_imp = min_imp
        self.min_imp_timeout = min_imp_timeout
        self.pbar = tqdm(desc='Testing tuples', total=self.n_tuples)

    def optimize(self):
        """
        Optimize alphas using Basin Hopping.

        Returns
        -------
        dict: {'best_tuple': self.best_tuple, 'best_O_R': self.best_O_R}
        """
        result = basinhopping(self.get_O_R_avg, self.guess, niter=self.n_tuples, niter_success=self.timeout,
                              callback=self.check_min_imp, accept_test=self.acceptable)

        self.best_O_R_avg = result.fun
        self.best_tuple = result.x
        self.best_O_R = self.get_O_R(result.x)

        if self.pbar:
            self.pbar.set_description('Completed')
            self.pbar.close()

        return {'best_tuple': self.best_tuple, 'best_O_R_avg': self.best_O_R_avg}

    def acceptable(self, f_new, x_new, f_old, x_old):
        return f_new <= self.best_O_R_avg and self.acceptable_alphas(x_new)

    def check_min_imp(self, x, O_R_avg, accepted) -> bool:
        if not accepted:
            return False

        if self.best_O_R_avg - O_R_avg < self.min_imp:
            self.failed_iterations += 1
            self.pbar.set_postfix(best_O_R=self.best_O_R_avg,
                                  fails=f'{self.failed_iterations}/{self.min_imp_timeout}')

            if self.failed_iterations >= self.min_imp_timeout:
                self.pbar.set_description('Min Improvement timeout reached')
                self.pbar.close()
                return True
        else:
            self.failed_iterations = 0

        self.best_O_R_avg = O_R_avg
        self.pbar.update()

    @staticmethod
    def acceptable_alphas(alphas):
        """
        Check if all alphas are in [0, 1] range
        Parameters
        ----------
        alphas: list
            Alphas to be checked

        Returns
        -------
        bool: True if all alphas are in [0, 1] range, False otherwise
        """
        for alpha in alphas:
            if not 0 <= alpha <= 1:
                return False
        return True
