import numpy as np
from pymoo.core.sampling import Sampling

class MGCVRPSampling(Sampling):
    def __init__(self, initial_solutions):
        super().__init__()
        self.initial_solutions = initial_solutions


    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)
        for i in range(n_samples):
            X[i, 0] = self.initial_solutions[i]
        return X